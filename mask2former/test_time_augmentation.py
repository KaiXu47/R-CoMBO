# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
from itertools import count

import numpy as np
import torch
from fvcore.transforms import HFlipTransform
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from detectron2.data.detection_utils import read_image
from detectron2.modeling import DatasetMapperTTA
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom


__all__ = [
    "SemanticSegmentorWithTTA",
]


class SemanticSegmentorWithTTA(nn.Module):
    """
    A SemanticSegmentor with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`SemanticSegmentor.forward`.
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=1):
        """
        Args:
            cfg (CfgNode):
            model (SemanticSegmentor): a SemanticSegmentor to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        self.cfg = cfg.clone()

        self.model = model

        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`SemanticSegmentor.forward`
        """

        def _maybe_read_image(dataset_dict):
            ret = copy.copy(dataset_dict)
            if "image" not in ret:
                image = read_image(ret.pop("file_name"), self.model.input_format)
                image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))  # CHW
                ret["image"] = image
            if "height" not in ret and "width" not in ret:
                ret["height"] = image.shape[1]
                ret["width"] = image.shape[2]
            return ret

        processed_results = []
        for x in batched_inputs:
            result = self._inference_one_image(_maybe_read_image(x))
            processed_results.append(result)
        return processed_results

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor
        Returns:
            dict: one output dict
        """
        if self.model.semantic_on and not self.model.panoptic_on and not self.model.instance_on:
            return self._inference_one_image_semantic(input)

        orig_shape = (input["height"], input["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(input)
        augmented_inputs, tfms = self._filter_panoptic_tta_inputs(augmented_inputs, tfms)

        final_cls = None
        final_masks = None
        count_predictions = 0
        for input, tfm in zip(augmented_inputs, tfms):
            count_predictions += 1
            with torch.no_grad():
                mask_cls, mask_pred = self._inference_raw(input, orig_shape)
                if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                    mask_pred = mask_pred.flip(dims=[2])

                if final_cls is None:
                    final_cls = mask_cls
                    final_masks = mask_pred
                else:
                    final_masks += mask_pred

        final_masks = final_masks / count_predictions

        result = {}
        if self.model.semantic_on:
            result["sem_seg"] = retry_if_cuda_oom(self.model.semantic_inference)(final_cls, final_masks)
        if self.model.panoptic_on:
            result["panoptic_seg"] = retry_if_cuda_oom(self.model.panoptic_inference)(final_cls, final_masks)
        if self.model.instance_on:
            result["instances"] = retry_if_cuda_oom(self.model.instance_inference)(final_cls, final_masks)
        return result

    def _inference_one_image_semantic(self, input):
        augmented_inputs, tfms = self._get_augmented_inputs(input)

        final_predictions = None
        count_predictions = 0
        for input, tfm in zip(augmented_inputs, tfms):
            count_predictions += 1
            with torch.no_grad():
                sem_seg = self.model([input])[0]["sem_seg"]
                if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                    sem_seg = sem_seg.flip(dims=[2])

                if final_predictions is None:
                    final_predictions = sem_seg
                else:
                    final_predictions += sem_seg

        return {"sem_seg": final_predictions / count_predictions}

    def _inference_raw(self, input, output_shape):
        image = input["image"].to(self.model.device)
        image = (image - self.model.pixel_mean) / self.model.pixel_std
        images = ImageList.from_tensors([image], self.model.size_divisibility)

        features = self.model.backbone(images.tensor)
        outputs = self.model.sem_seg_head(features)
        mask_cls = outputs["pred_logits"][0]
        mask_pred = outputs["pred_masks"]
        mask_pred = F.interpolate(
            mask_pred,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )[0]

        image_size = images.image_sizes[0]
        mask_pred = retry_if_cuda_oom(sem_seg_postprocess)(
            mask_pred,
            image_size,
            output_shape[0],
            output_shape[1],
        )
        mask_cls = mask_cls.to(mask_pred)
        return mask_cls, mask_pred

    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms

    def _filter_panoptic_tta_inputs(self, augmented_inputs, tfms):
        target_size = self.cfg.INPUT.MIN_SIZE_TEST
        kept_inputs = []
        kept_tfms = []
        for aug_input, tfm in zip(augmented_inputs, tfms):
            image = aug_input["image"]
            short_edge = min(image.shape[-2:])
            if short_edge == target_size:
                kept_inputs.append(aug_input)
                kept_tfms.append(tfm)

        if kept_inputs:
            return kept_inputs, kept_tfms
        return augmented_inputs[:1], tfms[:1]
