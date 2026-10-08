# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom


from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher


@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        current_catagory_ids: list = None,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        stuff_classes = getattr(metadata, "stuff_classes", [])
        self.background_class_id = (
            0 if len(stuff_classes) > 0 and stuff_classes[0].lower() == "background" else None
        )

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        if current_catagory_ids is not None:
            self.current_catagory_ids = torch.tensor(current_catagory_ids)

    def _strip_voc_future_proxy_logits(self, mask_cls):
        predictor = getattr(self.sem_seg_head, "predictor", None)
        if predictor is None or not getattr(predictor, "use_voc_future_proxy", False):
            return mask_cls

        expected_logits = sum(cls_embed.out_features for cls_embed in predictor.class_embeds)
        if mask_cls.shape[-1] != expected_logits:
            return mask_cls

        proxy_idx = getattr(predictor, "base_cls", None)
        if proxy_idx is None or proxy_idx < 0 or proxy_idx >= mask_cls.shape[-1] - 1:
            return mask_cls

        return torch.cat([mask_cls[..., :proxy_idx], mask_cls[..., proxy_idx + 1 :]], dim=-1)

    @classmethod
    def from_config(cls, cfg):
        is_voc_dataset = bool(cfg.DATASETS.TRAIN) and "pascal" in cfg.DATASETS.TRAIN[0]
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        weight_dict["loss_med_tokens"] = cfg.CONT.MED_TOKENS_WEIGHT
        weight_dict["loss_ce_distill"] = cfg.CONT.LOGIT_DISTILL_WEIGHT
        weight_dict["loss_feats"] = cfg.CONT.FEATURE_DISTILL_WEIGHT
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        num_tasks = 1 + (cfg.CONT.TOT_CLS - cfg.CONT.BASE_CLS) // cfg.CONT.INC_CLS
        n_cls_in_tasks = [cfg.CONT.BASE_CLS] + [cfg.CONT.INC_CLS] * (num_tasks - 1)
        current_start = sum(n_cls_in_tasks[:cfg.CONT.TASK - 1])
        current_end = sum(n_cls_in_tasks[:cfg.CONT.TASK])
        current_catagory_ids = list(range(current_start, current_end))
        if is_voc_dataset and (cfg.CONT.VOC_FUTURE_PROXY or cfg.CONT.VOC_FUTURE_PROXY_INIT) and cfg.CONT.TASK == 1:
            current_catagory_ids.append(current_end)

        criterion = SetCriterion(
            # sem_seg_head.num_classes,
            sum([cls_embed.out_features for cls_embed in sem_seg_head.predictor.class_embeds]) - 1,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            current_catagory_ids=current_catagory_ids,
            current_task=cfg.CONT.TASK,
            background_class_id=(
                0
                if len(getattr(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), "stuff_classes", [])) > 0
                and getattr(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), "stuff_classes", [])[0].lower() == "background"
                else None
            ),
            old_class_weight=cfg.CONT.VOC_REPLAY_OLD_CLASS_WEIGHT,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "current_catagory_ids": current_catagory_ids,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, old_pred=None, psd_label=False, old_med_tokens=None, 
                 flag=False, old_logits=None, old_aux_outputs=None, old_mask_features=None):
        """
        Args:
            importance:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        if "padding_size" in batched_inputs[0]:
            padding_sizes = [x["padding_size"] for x in batched_inputs]
        else:
            padding_sizes = range(len(batched_inputs))

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        med_tokens = outputs.pop("med_tokens")
        if self.training and flag == False:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            if old_pred is not None:
                psd_targets, old_targets = self.generate_psd_targets(targets, old_pred)

                # bipartite matching-based loss
                losses = self.criterion(outputs, targets, psd_targets, old_targets, med_tokens, old_med_tokens,  flag, old_logits, old_aux_outputs, old_mask_features)
            else:
                losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                    # losses[k] *= 10
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses

        elif self.training and flag == True:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            losses = self.criterion(outputs, targets, flag=flag)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            return losses

        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            aux_outputs = outputs["aux_outputs"]
            mask_features = outputs["mask_features"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size, padding_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes, padding_sizes
            ):
                mask_cls_result = self._strip_voc_future_proxy_logits(mask_cls_result)

                if psd_label:
                    output_psd_label = {}

                    scores, labels = F.softmax(mask_cls_result, dim=-1).max(-1)
                    n_cls = mask_cls_result.shape[-1] - 1
                    keep = labels.ne(n_cls)

                    sort = scores[keep].argsort(descending=True)

                    output_psd_label['labels'] = labels[keep][sort]
                    output_psd_label['masks'] = mask_pred_result[keep][sort].sigmoid()
                    output_psd_label['scores'] = scores[keep][sort]

                    processed_results.append(output_psd_label)
                else:
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

            if psd_label:
                return processed_results, med_tokens, mask_cls_results, aux_outputs, mask_features
            else:
                return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def generate_psd_targets(self, gt_targets, old_preds):
        if self.current_catagory_ids is not None:
            memory_part = []
            for tgt in gt_targets:
                labels = tgt["labels"]
                current_ids = self.current_catagory_ids.to(labels.device)
                is_current = torch.isin(labels, current_ids)
                if self.background_class_id is not None:
                    is_current = is_current | (labels == self.background_class_id)
                memory_part.append(bool(torch.logical_not(is_current).sum() != 0))
        else:
            memory_part = None

        psd_targets = []
        old_targets = []

        for i, (gt_target, old_pred) in enumerate(zip(gt_targets, old_preds)):
            psd_target = {}
            old_target = {}

            old_masks = old_pred["masks"]
            old_labels = old_pred["labels"]
            old_scores = old_pred["scores"]

            if self.current_catagory_ids is not None and old_labels.numel() > 0:
                old_class_end = int(self.current_catagory_ids[0].item())
                old_class_keep = old_labels < old_class_end
                if self.background_class_id is not None:
                    old_class_keep = old_class_keep & (old_labels != self.background_class_id)
                old_masks = old_masks[old_class_keep]
                old_labels = old_labels[old_class_keep]
                old_scores = old_scores[old_class_keep]

            if old_masks.shape[0] == 0:
                psd_target["labels"] = old_labels
                psd_target["masks"] = (old_masks >= 0.5)
                psd_targets.append(psd_target)
                old_target["labels"] = old_labels
                old_target["masks"] = (old_masks >= 0.5)
                old_targets.append(old_target)
                continue

            old_area = (old_masks >= 0.5).sum(dim=(1, 2))
            prob_masks = old_scores.view(-1, 1, 1) * old_masks
            mask_ids = prob_masks.argmax(0)

            non_ol_masks = torch.zeros_like(old_masks).to(torch.bool)
            for k in range(old_labels.shape[0]):
                non_ol_masks[k] = (mask_ids == k) & (old_masks[k] >= 0.5)

            if not memory_part[i]:
                if self.background_class_id is not None:
                    gt_foreground = gt_target["labels"] != self.background_class_id
                    if gt_foreground.any():
                        gt_region = torch.clamp(gt_target["masks"][gt_foreground].sum(0).int(), min=0, max=1)
                    elif gt_target["masks"].shape[0] > 0:
                        gt_region = torch.zeros_like(gt_target["masks"][0], dtype=torch.int)
                    else:
                        gt_region = torch.zeros_like(old_masks[0], dtype=torch.int)
                else:
                    if gt_target["masks"].shape[0] > 0:
                        gt_region = torch.clamp(gt_target["masks"].sum(0).int(), min=0, max=1)
                    else:
                        gt_region = torch.zeros_like(old_masks[0], dtype=torch.int)
                non_ol_masks = torch.logical_and(non_ol_masks, torch.logical_not(gt_region))

            new_area = non_ol_masks.sum(dim=(1, 2))
            keep = (new_area > 0) & (new_area > old_area * self.overlap_threshold)

            if self.background_class_id is not None and gt_target["labels"].numel() > 0:
                bg_ids = gt_target["labels"] == self.background_class_id
                if bg_ids.any():
                    old_union = (
                        non_ol_masks[keep].any(dim=0)
                        if keep.any()
                        else torch.zeros_like(gt_target["masks"][0], dtype=torch.bool)
                    )
                    gt_masks = gt_target["masks"].clone()
                    gt_masks[bg_ids] = torch.logical_and(gt_masks[bg_ids].bool(), torch.logical_not(old_union))
                    valid_gt = torch.logical_or(
                        gt_target["labels"] != self.background_class_id,
                        gt_masks.flatten(1).sum(dim=1) > 0,
                    )
                    gt_target["labels"] = gt_target["labels"][valid_gt]
                    gt_target["masks"] = gt_masks[valid_gt]

            psd_target["labels"] = old_labels[keep]
            psd_target["masks"] = non_ol_masks[keep]
            psd_targets.append(psd_target)

            old_target["labels"] = old_labels[keep]
            old_target["masks"] = old_masks[keep]
            old_targets.append(old_target)

        return psd_targets, old_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = self._strip_voc_future_proxy_logits(mask_cls)
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        mask_cls = self._strip_voc_future_proxy_logits(mask_cls)
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        # mask_pred = mask_pred.softmax(0)

        n_cls = mask_cls.shape[-1] - 1
        keep = labels.ne(n_cls) & (scores > self.object_mask_threshold)

        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        mask_cls = self._strip_voc_future_proxy_logits(mask_cls)

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        n_cls = mask_cls.shape[-1] - 1
        labels = torch.arange(n_cls, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        # topk_indices = topk_indices // self.sem_seg_head.num_classes
        topk_indices = topk_indices // n_cls
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def freeze_for_prompt_tuning(self):
        """Keep the final incremental model's backbone and shared decoder trainable."""
        return
