# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import resource

from mask2former.data.datasets.register_pascal_voc_21_semantic import register_all_pascal21, \
        register_current_voc_semantic

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import os
import math
import copy
from pathlib import Path

import numpy as np
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import verify_results
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import add_maskformer2_config
from mask2former.data.datasets.register_ade20k_panoptic import (
    register_current_ade20k_panoptic,
    register_complete_ade20k_sem,
)
from mask2former.data.datasets.register_ade20k_instance import register_current_ade20k_instance
from mask2former.data.datasets.register_cityscapes_panoptic import (
    CITYSCAPES_CATEGORIES,
    load_cityscapes_panoptic,
)

from continual import add_continual_config, Trainer
from continual.replay_buffer import build_voc_replay_buffer, load_voc_replay_buffer


def _resolve_step_output_dir(output_dir, task):
    output_path = Path(output_dir)
    step_dir = output_path.parent / f"step{task}"
    return step_dir


def _resolve_checkpoint_for_task(output_dir, task):
    step_dir = _resolve_step_output_dir(output_dir, task)
    final_ckpt = step_dir / "model_final.pth"
    if final_ckpt.exists():
        return str(final_ckpt)

    last_ckpt_file = step_dir / "last_checkpoint"
    if last_ckpt_file.exists():
        ckpt_name = last_ckpt_file.read_text().strip()
        if ckpt_name:
            candidate = step_dir / ckpt_name
            if candidate.exists():
                return str(candidate)

    raise FileNotFoundError(f"No usable checkpoint found under {step_dir}")




def _get_cityscapes_contiguous_metadata():
    meta = {
        "thing_classes": [k["name"] for k in CITYSCAPES_CATEGORIES],
        "thing_colors": [k["color"] for k in CITYSCAPES_CATEGORIES],
        "stuff_classes": [k["name"] for k in CITYSCAPES_CATEGORIES],
        "stuff_colors": [k["color"] for k in CITYSCAPES_CATEGORIES],
        "thing_dataset_id_to_contiguous_id": {},
        "stuff_dataset_id_to_contiguous_id": {},
    }
    for k in CITYSCAPES_CATEGORIES:
        train_id = k["trainId"]
        if k["isthing"] == 1:
            meta["thing_dataset_id_to_contiguous_id"][train_id] = train_id
        else:
            meta["stuff_dataset_id_to_contiguous_id"][train_id] = train_id
    return meta


def _register_current_cityscapes_panoptic(predefined_split):
    metadata = _get_cityscapes_contiguous_metadata()
    for prefix, (image_root, panoptic_root, panoptic_json) in predefined_split.items():
        DatasetCatalog.register(
            prefix,
            lambda x=image_root, y=panoptic_root, z=panoptic_json: load_cityscapes_panoptic(x, y, z, metadata),
        )
        MetadataCatalog.get(prefix).set(
            panoptic_root=panoptic_root,
            image_root=image_root,
            panoptic_json=panoptic_json,
            gt_dir=panoptic_root.replace("cityscapes_panoptic_", ""),
            sem_seg_root=panoptic_root.replace("cityscapes_panoptic_", ""),
            evaluator_type="cityscapes_panoptic_seg",
            ignore_label=255,
            label_divisor=1000,
            **metadata,
        )


def _register_current_semantic_from_panoptic(train_name, val_name, pan_train_name, pan_val_name):
    current_sem_data_train = DatasetCatalog.get(pan_train_name)
    current_sem_data_val = DatasetCatalog.get(pan_val_name)
    complete_sem_meta_train = MetadataCatalog.get(pan_train_name)
    complete_sem_meta_val = MetadataCatalog.get(pan_val_name)

    DatasetCatalog.register(train_name, lambda: current_sem_data_train)
    DatasetCatalog.register(val_name, lambda: current_sem_data_val)
    MetadataCatalog.get(train_name).set(
        evaluator_type="sem_seg",
        ignore_label=complete_sem_meta_train.ignore_label,
        image_root=complete_sem_meta_train.image_root,
        sem_seg_root=getattr(complete_sem_meta_train, "sem_seg_root", None),
        stuff_classes=complete_sem_meta_train.stuff_classes,
        stuff_colors=complete_sem_meta_train.stuff_colors,
        stuff_dataset_id_to_contiguous_id=complete_sem_meta_train.stuff_dataset_id_to_contiguous_id,
    )
    MetadataCatalog.get(val_name).set(
        evaluator_type="sem_seg",
        ignore_label=complete_sem_meta_val.ignore_label,
        image_root=complete_sem_meta_val.image_root,
        sem_seg_root=getattr(complete_sem_meta_val, "sem_seg_root", None),
        stuff_classes=complete_sem_meta_val.stuff_classes,
        stuff_colors=complete_sem_meta_val.stuff_colors,
        stuff_dataset_id_to_contiguous_id=complete_sem_meta_val.stuff_dataset_id_to_contiguous_id,
    )


def _resolve_voc_incremental_split(filename):
    project_root = Path(__file__).resolve().parent
    candidates = []

    if os.path.isabs(filename):
        candidates.append(Path(filename))
    else:
        candidates.append(Path(filename))
        candidates.append(Path("incremental_split") / filename)
        candidates.append(project_root / "incremental_split" / filename)

    split_root = os.getenv("VOC_INCREMENTAL_SPLIT_ROOT")
    if split_root:
        candidates.append(Path(split_root) / filename)

    dataset_root = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    candidates.extend(
        [
            dataset_root / "pascal_voc_d2" / "incremental_split" / filename,
            dataset_root / "voc" / "incremental_split" / filename,
        ]
    )

    for path in candidates:
        if path.exists():
            return str(path.resolve())

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot find VOC incremental split {filename}. Searched:\n{searched}")


_VOC_SIMILAR_OLD_CLASSES = {
    16: [5, 9],       # pottedplant: bottle/chair
    17: [10, 12, 13], # sheep: cow/dog/horse
    18: [9, 11],      # sofa: chair/diningtable
    19: [6, 7],       # train: bus/car
    20: [11, 9],      # tvmonitor: diningtable/chair
}


def _voc_current_class_ids(cfg):
    if cfg.CONT.TASK <= 1:
        return list(range(0, cfg.CONT.BASE_CLS))
    start = cfg.CONT.BASE_CLS + (cfg.CONT.TASK - 2) * cfg.CONT.INC_CLS
    end = min(cfg.CONT.BASE_CLS + (cfg.CONT.TASK - 1) * cfg.CONT.INC_CLS, cfg.CONT.TOT_CLS)
    return list(range(start, end))


def _voc_boundary_replay_classes(cfg):
    if cfg.CONT.TASK <= 1:
        return []
    old_class_end = _voc_current_class_ids(cfg)[0]
    replay_classes = []
    for new_id in _voc_current_class_ids(cfg):
        replay_classes.extend(
            old_id for old_id in _VOC_SIMILAR_OLD_CLASSES.get(new_id, []) if 0 < old_id < old_class_end
        )
    return sorted(set(replay_classes))


def _add_voc_boundary_replay(current_data, complete_data, cfg):
    if (
        not cfg.CONT.VOC_BOUNDARY_REPLAY
        or cfg.CONT.TASK <= 1
        or cfg.CONT.VOC_BOUNDARY_REPLAY_MAX_IMAGES_PER_CLASS <= 0
    ):
        return current_data

    replay_classes = _voc_boundary_replay_classes(cfg)
    if not replay_classes:
        return current_data

    max_per_class = int(cfg.CONT.VOC_BOUNDARY_REPLAY_MAX_IMAGES_PER_CLASS)
    current_files = {x["file_name"] for x in current_data}
    added_per_class = {cid: 0 for cid in replay_classes}
    replay_data = []

    for data in complete_data:
        if data["file_name"] in current_files:
            continue
        sem_seg_file = data.get("sem_seg_file_name")
        if sem_seg_file is None or not os.path.exists(sem_seg_file):
            continue
        labels = set(np.unique(np.array(Image.open(sem_seg_file))).astype(int).tolist())
        matched = [cid for cid in replay_classes if cid in labels and added_per_class[cid] < max_per_class]
        if not matched:
            continue

        replay_item = copy.deepcopy(data)
        replay_item["voc_boundary_replay"] = True
        # The semantic mapper dispatches replay samples via ``voc_replay``.
        # Keep the boundary-specific marker for diagnostics, while also setting
        # the common marker so these samples actually take the replay path.
        replay_item["voc_replay"] = True
        replay_item["voc_replay_classes"] = sorted(set(matched))
        replay_data.append(replay_item)
        for cid in matched:
            added_per_class[cid] += 1

        if all(v >= max_per_class for v in added_per_class.values()):
            break

    print(
        "[VOC boundary replay] "
        f"task={cfg.CONT.TASK}, replay_classes={replay_classes}, "
        f"added={len(replay_data)}, per_class={added_per_class}"
    )
    return current_data + replay_data


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_continual_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if cfg.CONT.WEIGHTS is not None:
        cfg.MODEL.WEIGHTS = cfg.CONT.WEIGHTS
    elif cfg.CONT.TASK > 1:
        if args.eval_only:
            cfg.MODEL.WEIGHTS = _resolve_checkpoint_for_task(cfg.OUTPUT_DIR, cfg.CONT.TASK)
        else:
            cfg.MODEL.WEIGHTS = _resolve_checkpoint_for_task(cfg.OUTPUT_DIR, cfg.CONT.TASK - 1)

        if cfg.CONT.OLD_WEIGHTS is None:
            cfg.CONT.OLD_WEIGHTS = cfg.MODEL.WEIGHTS

    elif args.eval_only:
        cfg.MODEL.WEIGHTS = _resolve_checkpoint_for_task(cfg.OUTPUT_DIR, cfg.CONT.TASK)

    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)

    # panoptic segmentation, or semantic segmentation that reuses a panoptic image split
    uses_panoptic_split = (
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON == True
        or (
            cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON == False
            and cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON == False
            and (
                "ade20k" in cfg.DATASETS.TRAIN[0]
                or "cityscapes" in cfg.DATASETS.TRAIN[0]
            )
        )
    )
    if uses_panoptic_split:
        if "cityscapes" in cfg.DATASETS.TRAIN[0]:
            city_root = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "cityscapes")
            predefined_split = {
                "current_cityscapes_panoptic_train": (
                    os.path.join(city_root, "leftImg8bit/train"),
                    os.path.join(city_root, "gtFine/cityscapes_panoptic_train"),
                    f"json/cityscapes_pan/train_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK}_pan.json",
                ),
                "current_cityscapes_panoptic_val": (
                    os.path.join(city_root, "leftImg8bit/val"),
                    os.path.join(city_root, "gtFine/cityscapes_panoptic_val"),
                    f"json/cityscapes_pan/val_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK}_pan.json",
                ),
            }
            _register_current_cityscapes_panoptic(predefined_split)
        else:
            predefined_split = {
                "current_ade20k_panoptic_train": (
                    "datasets/ADEChallengeData2016/images/training",
                    "datasets/ADEChallengeData2016/ade20k_panoptic_train",
                    f"json/pan/train_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK}_pan.json",
                    "datasets/ADEChallengeData2016/annotations_detectron2/training",
                    f"json/pan/train_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK}_inst.json",
                ),
                "current_ade20k_panoptic_val": (
                    "datasets/ADEChallengeData2016/images/validation",
                    "datasets/ADEChallengeData2016/ade20k_panoptic_val",
                    f"json/pan/val_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK}_pan.json",
                    "datasets/ADEChallengeData2016/annotations_detectron2/validation",
                    f"json/pan/val_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK}_inst.json",
                ),
            }
            register_current_ade20k_panoptic(predefined_split)

    if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON == False and cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON == False:

        if "cityscapes" in cfg.DATASETS.TRAIN[0]:
            _register_current_semantic_from_panoptic(
                "current_cityscapes_sem_seg_train",
                "current_cityscapes_sem_seg_val",
                "current_cityscapes_panoptic_train",
                "current_cityscapes_panoptic_val",
            )
        elif "ade20k" in cfg.DATASETS.TRAIN[0]:
            register_complete_ade20k_sem("datasets")
            complete_sem_data_train = DatasetCatalog.get("complete_ade20k_sem_seg_train")
            complete_sem_data_val = DatasetCatalog.get("complete_ade20k_sem_seg_val")
            img_list_train = [img['file_name'] for img in DatasetCatalog.get("current_ade20k_panoptic_train")]
            img_list_val = [img['file_name'] for img in DatasetCatalog.get("current_ade20k_panoptic_val")]
            current_sem_data_train = []
            current_sem_data_val = []
            for img in complete_sem_data_train:
                if img['file_name'] in img_list_train:
                    current_sem_data_train.append(img)
            current_sem_data_train = _add_voc_boundary_replay(current_sem_data_train, complete_sem_data_train, cfg)
            for img in complete_sem_data_val:
                if img['file_name'] in img_list_val:
                    current_sem_data_val.append(img)
            complete_sem_meta_train = MetadataCatalog.get("complete_ade20k_sem_seg_train")
            complete_sem_meta_val = MetadataCatalog.get("complete_ade20k_sem_seg_val")

            DatasetCatalog.register(
                "current_ade20k_sem_seg_train", lambda: current_sem_data_train
            )
            DatasetCatalog.register(
                "current_ade20k_sem_seg_val", lambda: current_sem_data_val
            )
            MetadataCatalog.get("current_ade20k_sem_seg_train").set(
                evaluator_type=complete_sem_meta_train.evaluator_type,
                ignore_label=complete_sem_meta_train.ignore_label,
                image_root=complete_sem_meta_train.image_root,
                sem_seg_root=complete_sem_meta_train.sem_seg_root,
                stuff_classes=complete_sem_meta_train.stuff_classes,
            )
            MetadataCatalog.get("current_ade20k_sem_seg_val").set(
                evaluator_type=complete_sem_meta_val.evaluator_type,
                ignore_label=complete_sem_meta_val.ignore_label,
                image_root=complete_sem_meta_val.image_root,
                sem_seg_root=complete_sem_meta_val.sem_seg_root,
                stuff_classes=complete_sem_meta_val.stuff_classes,
            )
        else:
            voc_root = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "pascal_voc_d2")
            voc_split_name = f"{cfg.CONT.BASE_CLS - 1}-{cfg.CONT.INC_CLS}"
            predefined_split = {
                "current_pascal21_sem_seg_train1": (
                    os.path.join(voc_root, "images/training"),
                    os.path.join(voc_root, "annotations_pascal21/training"),
                    _resolve_voc_incremental_split(f"train_{voc_split_name}_step_{cfg.CONT.TASK}.txt"),
                ),
                "current_pascal21_sem_seg_val1": (
                    os.path.join(voc_root, "images/validation"),
                    os.path.join(voc_root, "annotations_pascal21/validation"),
                    _resolve_voc_incremental_split(f"val_{voc_split_name}_step_{cfg.CONT.TASK}.txt"),
                ),
            }
            register_current_voc_semantic(predefined_split)

            complete_sem_data_train = DatasetCatalog.get("openvocab_pascal21_sem_seg_train")
            complete_sem_data_val = DatasetCatalog.get("openvocab_pascal21_sem_seg_val")
            img_list_train = [img['file_name'] for img in DatasetCatalog.get("current_pascal21_sem_seg_train1")]
            img_list_val = [img['file_name'] for img in DatasetCatalog.get("current_pascal21_sem_seg_val1")]
            current_sem_data_train = []
            current_sem_data_val = []
            for img in complete_sem_data_train:
                if img['file_name'] in img_list_train:
                    current_sem_data_train.append(img)
            # ---- VOC replay buffer: load samples from previous steps ----
            if cfg.CONT.VOC_BOUNDARY_REPLAY and cfg.CONT.TASK > 1:
                prev_step_dir = _resolve_step_output_dir(cfg.OUTPUT_DIR, cfg.CONT.TASK - 1)
                replay_path = os.path.join(str(prev_step_dir), "replay_buffer.json")
                replay_images = load_voc_replay_buffer(replay_path)
                # Oversample replay images to counter long-tail imbalance
                ratio = int(cfg.CONT.VOC_REPLAY_SAMPLE_RATIO)
                if ratio > 1:
                    replay_images = replay_images * ratio
                current_sem_data_train = current_sem_data_train + replay_images
            for img in complete_sem_data_val:
                if img['file_name'] in img_list_val:
                    current_sem_data_val.append(img)
            complete_sem_meta_train = MetadataCatalog.get("openvocab_pascal21_sem_seg_train")
            complete_sem_meta_val = MetadataCatalog.get("openvocab_pascal21_sem_seg_val")

            DatasetCatalog.register(
                "current_pascal21_sem_seg_train", lambda: current_sem_data_train
            )
            DatasetCatalog.register(
                "current_pascal21_sem_seg_val", lambda: current_sem_data_val
            )
            MetadataCatalog.get("current_pascal21_sem_seg_train").set(
                evaluator_type=complete_sem_meta_train.evaluator_type,
                ignore_label=complete_sem_meta_train.ignore_label,
                image_root=complete_sem_meta_train.image_root,
                sem_seg_root=complete_sem_meta_train.sem_seg_root,
                stuff_classes=complete_sem_meta_train.stuff_classes,
            )
            MetadataCatalog.get("current_pascal21_sem_seg_val").set(
                evaluator_type=complete_sem_meta_val.evaluator_type,
                ignore_label=complete_sem_meta_val.ignore_label,
                image_root=complete_sem_meta_val.image_root,
                sem_seg_root=complete_sem_meta_val.sem_seg_root,
                stuff_classes=complete_sem_meta_val.stuff_classes,
            )

    if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON == False and cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON == False:

        predefined_split = {
            "current_ade20k_instance_train": (
                "datasets/ADEChallengeData2016/images/training",
                f"json/inst/train_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK}_inst.json",
            ),
            "current_ade20k_instance_val": (
                "datasets/ADEChallengeData2016/images/validation",
                f"json/inst/val_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK}_inst.json",
            ),
        }
        register_current_ade20k_instance(predefined_split)

        if cfg.CONT.TASK > 1 and cfg.CONT.MEMORY == True:
            predefined_split_memory = {
                "memory_ade20k_instance_train": (
                    "datasets/ADEChallengeData2016/images/training",
                    f"json/memory/inst/train_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK - 1}_inst.json",
                ),
            }
            register_current_ade20k_instance(predefined_split_memory)

            current_inst_data_train = DatasetCatalog.get("current_ade20k_instance_train")
            current_inst_meta_train = MetadataCatalog.get("current_ade20k_instance_train")
            memory_inst_data_train = DatasetCatalog.get("memory_ade20k_instance_train")

            cur_w_mem_inst_data_train = current_inst_data_train + memory_inst_data_train
            DatasetCatalog.register(
                "cur_w_mem_ade20k_instance_train", lambda: cur_w_mem_inst_data_train
            )
            MetadataCatalog.get("cur_w_mem_ade20k_instance_train").set(
                evaluator_type=current_inst_meta_train.evaluator_type,
                image_root=current_inst_meta_train.image_root,
                # json_file=cur_w_mem_inst_meta_train.json_file,
                name="cur_w_mem_ade20k_instance_train",
                thing_classes=current_inst_meta_train.thing_classes,
                thing_dataset_id_to_contiguous_id=current_inst_meta_train.thing_dataset_id_to_contiguous_id,
            )
            cfg.defrost()
            cfg.DATASETS.TRAIN = ("cur_w_mem_ade20k_instance_train",)
            cfg.freeze()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    result = trainer.train()

    # ---- Build VOC replay buffer after training (before next step sees its data) ----
    if (
        cfg.CONT.VOC_BOUNDARY_REPLAY
        and comm.is_main_process()
        and "pascal" in cfg.DATASETS.TRAIN[0]
    ):
        num_tasks = 1 + (cfg.CONT.TOT_CLS - cfg.CONT.BASE_CLS) // cfg.CONT.INC_CLS
        n_cls_in_tasks = [cfg.CONT.BASE_CLS] + [cfg.CONT.INC_CLS] * (num_tasks - 1)
        all_class_ids = list(range(1, sum(n_cls_in_tasks[: cfg.CONT.TASK])))  # exclude background (0)

        current_train = DatasetCatalog.get("current_pascal21_sem_seg_train")
        complete_train = DatasetCatalog.get("openvocab_pascal21_sem_seg_train")
        current_files = {d["file_name"] for d in current_train}
        # Use complete GT (all classes visible) for fair class-area counting
        candidate_data = [d for d in complete_train if d["file_name"] in current_files]

        buffer_path = os.path.join(cfg.OUTPUT_DIR, "replay_buffer.json")
        buffer_capacity = int(cfg.CONT.VOC_BOUNDARY_REPLAY_MAX_IMAGES_PER_CLASS)
        build_voc_replay_buffer(candidate_data, all_class_ids, buffer_capacity, buffer_path)

    return result


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
