import copy
import itertools
import logging
import os
import weakref

from collections import OrderedDict
from tabulate import tabulate
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, get_detection_dataset_dicts
from detectron2.engine import (
    HookBase,
    TrainerBase,
    create_ddp_model,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.logger import setup_logger, _log_api_usage

from torch.utils.data import DistributedSampler, SequentialSampler

# MaskFormer
from mask2former import (
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

from .continual_panoptic_dataset_mapper import ContinualPanopticDatasetMapper
from .continual_semantic_dataset_mapper import ContinualSemanticDatasetMapper
from .continual_instance_dataset_mapper import ContinualInstanceDatasetMapper
from .evaluator import SemSegEvaluator, COCOPanopticEvaluator, InstanceSegEvaluator
from .train_loop import SimpleTrainer1, AMPTrainer1
from .Default_Trainer import DefaultTrainer

class Trainer(DefaultTrainer):

    def __init__(self, cfg):
        self._hooks: List[HookBase] = []
        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int
        self.storage: EventStorage
        _log_api_usage("trainer." + self.__class__.__name__)

        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()

        if cfg.CONT.TASK > 1 and cfg.CONT.USE_OLD_MODEL:
            cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
            cfg_old = cfg.clone()
            cfg_old.defrost()
            cfg_old.CONT.TASK = cfg.CONT.TASK - 1
            cfg_old.freeze()

            model_old = self.build_model(cfg_old).eval()
            checkpoint_state = torch.load(
                cfg.CONT.OLD_WEIGHTS, map_location=torch.device("cpu")
            )["model"]
            dynamic_importance_keys = {
                "criterion.query_importance",
                "criterion.query_importance_accumulator",
            }
            importance_state = {
                key: checkpoint_state.pop(key)
                for key in dynamic_importance_keys
                if key in checkpoint_state
            }
            model_old_state = model_old.state_dict()
            incompatible_keys = [
                key for key, value in checkpoint_state.items()
                if key in model_old_state and model_old_state[key].shape != value.shape
            ]
            for key in incompatible_keys:
                checkpoint_state.pop(key)
            if incompatible_keys:
                logger.warning(
                    "Skipping incompatible parameters when loading old model from %s: %s",
                    cfg.CONT.OLD_WEIGHTS,
                    incompatible_keys,
            )
            model_old.load_state_dict(checkpoint_state, strict=False)
            old_criterion = model_old.criterion if hasattr(model_old, "criterion") else None
            if old_criterion is not None:
                query_importance = importance_state.get("criterion.query_importance")
                if query_importance is not None and query_importance.numel() > 0:
                    old_criterion.query_importance = query_importance.detach().to(
                        old_criterion.empty_weight.device
                    ).clone()
                    logger.info(
                        "Loaded old query importance from %s with shape %s",
                        cfg.CONT.OLD_WEIGHTS,
                        tuple(old_criterion.query_importance.shape),
                    )

                query_importance_accumulator = importance_state.get(
                    "criterion.query_importance_accumulator"
                )
                if (
                    query_importance_accumulator is not None
                    and query_importance_accumulator.numel() > 0
                ):
                    old_criterion.query_importance_accumulator = (
                        query_importance_accumulator.detach()
                        .to(old_criterion.empty_weight.device)
                        .clone()
                    )
            if old_criterion is not None and not old_criterion.has_query_importance():
                old_criterion.finalize_query_importance()
            model_old = create_ddp_model(model_old, broadcast_buffers=False)
        else:
            model_old = None
            old_criterion = None

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        if cfg.CONT.TASK > 1 and cfg.CONT.USE_OLD_MODEL:
            if hasattr(model, "criterion") and old_criterion is not None:
                model.criterion.inherit_query_importance(old_criterion)
                if cfg.CONT.IMPORTANCE_GRAD_ROUTING:
                    predictor = getattr(getattr(model, "sem_seg_head", None), "predictor", None)
                    if predictor is None or not hasattr(predictor, "set_routing_query_importance"):
                        raise AttributeError(
                            "Importance routing requires a compatible transformer decoder"
                        )
                    predictor.set_routing_query_importance(model.criterion.query_importance)
                    logger.info(
                        "Enabled importance gradient routing with shape %s and gradient floor %.3f",
                        tuple(model.criterion.query_importance.shape),
                        cfg.CONT.IMPORTANCE_GRAD_MIN,
                    )
            model.freeze_for_prompt_tuning()
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)
        data_loader_finite = self.build_train_loader(cfg, sampler=DistributedSampler if torch.cuda.device_count() > 1 else SequentialSampler)

        model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=True)
        self._trainer = (AMPTrainer1 if cfg.SOLVER.AMP.ENABLED else SimpleTrainer1)(
            model, model_old, data_loader, optimizer, cfg, data_loader_finite=data_loader_finite
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def train(self):
        results = super().train()
        if self.cfg.TEST.AUG.ENABLED:
            model = getattr(self, "model", None)
            if model is None:
                model = self._trainer.model
            tta_results = self.test_with_TTA(self.cfg, model)
            if results is None:
                results = OrderedDict()
            results.update(tta_results)
        return results


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(
                SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder, cfg=cfg)
            )
        # panoptic segmentation
        if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder, cfg=cfg))
        # ADE20K
        # if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
        if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder, cfg=cfg))

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "continual_panoptic":
            mapper = ContinualPanopticDatasetMapper(cfg, True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "continual_semantic":
            mapper = ContinualSemanticDatasetMapper(cfg, True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "continual_instance":
            mapper = ContinualInstanceDatasetMapper(cfg, True)
        else:
            mapper = None

        if sampler is None:
            return build_detection_train_loader(cfg, mapper=mapper, sampler=sampler)
        else:
            dataset = get_detection_dataset_dicts(
                cfg.DATASETS.TRAIN,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                if cfg.MODEL.KEYPOINT_ON
                else 0,
                proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
            )
            _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])
            return build_detection_train_loader(
                dataset=dataset, mapper=mapper, sampler=sampler(dataset), total_batch_size=cfg.SOLVER.IMS_PER_BATCH, 
                aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING, num_workers=cfg.DATALOADER.NUM_WORKERS)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger("detectron2.trainer")
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

                table_list = []
                if "sem_seg" in results_i.keys():
                    table_list.append([
                        "mIoU",
                        results_i["sem_seg"]["mIoU_old"],
                        results_i["sem_seg"]["mIoU_new"],
                        results_i["sem_seg"]["mIoU_past"],
                        results_i["sem_seg"]["mIoU_current"],
                        results_i["sem_seg"]["mIoU_all"]
                    ])
                    table_list.append([
                        "mACC",
                        results_i["sem_seg"]["mACC_old"],
                        results_i["sem_seg"]["mACC_new"],
                        results_i["sem_seg"]["mACC_past"],
                        results_i["sem_seg"]["mACC_current"],
                        results_i["sem_seg"]["mACC_all"]
                    ])

                if "segm" in results_i.keys():
                    table_list.append([
                        "AP",
                        results_i["segm"]["AP_old"],
                        results_i["segm"]["AP_new"],
                        results_i["segm"]["AP_past"],
                        results_i["segm"]["AP_current"],
                        results_i["segm"]["AP_all"]
                    ])

                if "panoptic_seg" in results_i.keys():
                    table_list.append([
                        "PQ",
                        results_i["panoptic_seg"]["PQ_Old"],
                        results_i["panoptic_seg"]["PQ_New"],
                        results_i["panoptic_seg"]["PQ_Past"],
                        results_i["panoptic_seg"]["PQ_Current"],
                        results_i["panoptic_seg"]["PQ_All"]
                    ])
                    table_list.append([
                        "PQ_th",
                        results_i["panoptic_seg"]["PQ_Old_th"],
                        results_i["panoptic_seg"]["PQ_New_th"],
                        results_i["panoptic_seg"]["PQ_Past_th"],
                        results_i["panoptic_seg"]["PQ_Current_th"],
                        results_i["panoptic_seg"]["PQ_All_th"]
                    ])
                    table_list.append([
                        "PQ_st",
                        results_i["panoptic_seg"]["PQ_Old_st"],
                        results_i["panoptic_seg"]["PQ_New_st"],
                        results_i["panoptic_seg"]["PQ_Past_st"],
                        results_i["panoptic_seg"]["PQ_Current_st"],
                        results_i["panoptic_seg"]["PQ_All_st"]
                    ])
                    table_list.append([
                        "SQ",
                        results_i["panoptic_seg"]["SQ_Old"],
                        results_i["panoptic_seg"]["SQ_New"],
                        results_i["panoptic_seg"]["SQ_Past"],
                        results_i["panoptic_seg"]["SQ_Current"],
                        results_i["panoptic_seg"]["SQ_All"]
                    ])
                    table_list.append([
                        "SQ_th",
                        results_i["panoptic_seg"]["SQ_Old_th"],
                        results_i["panoptic_seg"]["SQ_New_th"],
                        results_i["panoptic_seg"]["SQ_Past_th"],
                        results_i["panoptic_seg"]["SQ_Current_th"],
                        results_i["panoptic_seg"]["SQ_All_th"]
                    ])
                    table_list.append([
                        "SQ_st",
                        results_i["panoptic_seg"]["SQ_Old_st"],
                        results_i["panoptic_seg"]["SQ_New_st"],
                        results_i["panoptic_seg"]["SQ_Past_st"],
                        results_i["panoptic_seg"]["SQ_Current_st"],
                        results_i["panoptic_seg"]["SQ_All_st"]
                    ])
                    table_list.append([
                        "RQ",
                        results_i["panoptic_seg"]["RQ_Old"],
                        results_i["panoptic_seg"]["RQ_New"],
                        results_i["panoptic_seg"]["RQ_Past"],
                        results_i["panoptic_seg"]["RQ_Current"],
                        results_i["panoptic_seg"]["RQ_All"]
                    ])
                    table_list.append([
                        "RQ_th",
                        results_i["panoptic_seg"]["RQ_Old_th"],
                        results_i["panoptic_seg"]["RQ_New_th"],
                        results_i["panoptic_seg"]["RQ_Past_th"],
                        results_i["panoptic_seg"]["RQ_Current_th"],
                        results_i["panoptic_seg"]["RQ_All_th"]
                    ])
                    table_list.append([
                        "RQ_st",
                        results_i["panoptic_seg"]["RQ_Old_st"],
                        results_i["panoptic_seg"]["RQ_New_st"],
                        results_i["panoptic_seg"]["RQ_Past_st"],
                        results_i["panoptic_seg"]["RQ_Current_st"],
                        results_i["panoptic_seg"]["RQ_All_st"]
                    ])

                table_headers = ["", "old", "new", "past", "current", "all"]
                table = tabulate(table_list, headers=table_headers,
                                 tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center")
                logger.info("Result Summary:\n" + table)

        if len(results) == 1:
            results = list(results.values())[0]
        return results
