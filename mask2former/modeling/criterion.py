# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule

def normalize(tensor, dim=-1, eps=1e-6):
    if tensor.numel() == 0 or tensor.shape[dim] == 0:
        return tensor
    min_val, max_val = tensor.min(dim=dim, keepdim=True)[0], tensor.max(dim=dim, keepdim=True)[0]
    return (tensor - min_val) / (max_val - min_val + eps)

def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, current_catagory_ids=None, current_task=None,
                 background_class_id=None, old_class_weight=1.0):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        memory_empty_weight = self.empty_weight.clone()
        memory_empty_weight[-1] = 0.0
        self.register_buffer("memory_empty_weight", memory_empty_weight)
        # Per-class CE weights: give old classes a boost to counter long-tail
        # imbalance between abundant new-class pixels and sparse replay old-class pixels.
        self.smooth_weights = torch.ones(self.num_classes)
        if old_class_weight != 1.0 and current_catagory_ids is not None and len(current_catagory_ids) > 0:
            old_class_end = int(current_catagory_ids[0])
            if old_class_end > 0:
                self.smooth_weights[:old_class_end] *= old_class_weight
        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.current_inc_task = current_task - 2
        self.current_catagory_ids = torch.tensor(current_catagory_ids)
        # self.empty_weight[:self.current_catagory_ids[0]] *= 0.4

        self.beta = len(current_catagory_ids) / (current_catagory_ids[-1] + 1)
        self.register_buffer("query_importance", torch.zeros(0, 0))
        self.register_buffer("query_importance_accumulator", torch.zeros(0, 0))
        self.background_class_id = background_class_id

    def _is_memory_sample(self, target):
        if self.current_catagory_ids is None:
            return False
        labels = target["labels"]
        current_ids = self.current_catagory_ids.to(labels.device)
        is_current = torch.isin(labels, current_ids)
        if self.background_class_id is not None:
            is_current = is_current | (labels == self.background_class_id)
        return bool(torch.logical_not(is_current).sum() != 0)

    def _ensure_importance_buffers(self, num_layers, num_queries, device):
        expected_shape = (num_layers, num_queries)
        if tuple(self.query_importance.shape) != expected_shape:
            self.query_importance = torch.zeros(expected_shape, device=device)
        if tuple(self.query_importance_accumulator.shape) != expected_shape:
            self.query_importance_accumulator = torch.zeros(expected_shape, device=device)

    def has_query_importance(self):
        return self.query_importance.numel() > 0 and torch.count_nonzero(self.query_importance).item() > 0

    def begin_query_importance_update(self):
        if self.query_importance_accumulator.numel() > 0:
            self.query_importance_accumulator.zero_()

    def finalize_query_importance(self):
        if self.query_importance_accumulator.numel() == 0 or self.query_importance_accumulator.shape[1] == 0:
            return

        with torch.no_grad():
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(self.query_importance_accumulator)

            accumulated_importance = normalize(self.query_importance_accumulator)
            accumulated_importance = 1.0 - accumulated_importance
            accumulated_importance = accumulated_importance / (
                accumulated_importance.mean(dim=1, keepdim=True) + 1e-6
            )

            if tuple(self.query_importance.shape) != tuple(accumulated_importance.shape):
                self.query_importance = torch.zeros_like(accumulated_importance)

            if self.has_query_importance():
                self.query_importance = (1 - self.beta) * self.query_importance + self.beta * accumulated_importance
            else:
                self.query_importance = accumulated_importance.clone()

            self.query_importance = self.query_importance / (
                self.query_importance.mean(dim=1, keepdim=True) + 1e-6
            )
            self.query_importance_accumulator.zero_()

    def inherit_query_importance(self, other_criterion):
        if other_criterion is None:
            return
        if other_criterion.query_importance.numel() > 0:
            self.query_importance = other_criterion.query_importance.detach().to(self.empty_weight.device).clone()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        legacy_key_map = {
            "class_weight": "query_importance",
            "class_temp_weight": "query_importance_accumulator",
        }
        for legacy_name, new_name in legacy_key_map.items():
            legacy_key = prefix + legacy_name
            new_key = prefix + new_name
            if legacy_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(legacy_key)

        # Ignore retired scalar training counters while preserving strict loading
        # for live buffers, nested modules, and unknown non-scalar tensors.
        current_keys = {prefix + key for key in self.state_dict()}
        for key in list(state_dict):
            if not key.startswith(prefix) or key in current_keys:
                continue
            local_key = key[len(prefix):]
            value = state_dict[key]
            if "." not in local_key and isinstance(value, torch.Tensor) and value.ndim == 0:
                state_dict.pop(key)

        for buffer_name in legacy_key_map.values():
            state_key = prefix + buffer_name
            if state_key in state_dict:
                loaded_tensor = state_dict[state_key]
                current_tensor = getattr(self, buffer_name)
                if tuple(current_tensor.shape) != tuple(loaded_tensor.shape):
                    setattr(self, buffer_name, torch.zeros_like(loaded_tensor))

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def loss_labels(self, outputs, targets, indices, num_masks, memory_part=None, hdhl=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o


        if memory_part is not None and sum(memory_part) > 0:
            non_memory_part = [not m for m in memory_part]
            memory_empty_weight = self.memory_empty_weight.clone()
            smooth_weights = self.smooth_weights.to(memory_empty_weight.device)
            memory_empty_weight[:-1] *= smooth_weights
            loss_ce_memory = F.cross_entropy(src_logits[memory_part].transpose(1, 2), target_classes[memory_part],
                                             memory_empty_weight, reduction='none')

            num_bg_mem = sum(memory_part) * src_logits.size(1) - \
                         sum([len(index[0]) for i, index in enumerate(indices) if memory_part[i]])
            num_bg_curr = sum(non_memory_part) * src_logits.size(1) - \
                          sum([len(index[0]) for i, index in enumerate(indices) if non_memory_part[i]])
            empty_weight = self.empty_weight.clone()
            if num_bg_curr > 0:
                empty_weight[-1] *= (num_bg_mem + num_bg_curr) / num_bg_curr
            smooth_weights = self.smooth_weights.to(empty_weight.device)
            empty_weight[:-1] *= smooth_weights
            loss_ce = F.cross_entropy(
                src_logits[non_memory_part].transpose(1, 2), target_classes[non_memory_part], empty_weight,
                reduction='none')

            weights_for_empty_weight = src_logits.size(0) * src_logits.size(1) * 0.1 + \
                                       sum([len(index[0]) for i, index in enumerate(indices)]) * 0.9

            loss_ce = (loss_ce_memory.sum() + loss_ce.sum()) / weights_for_empty_weight
        else:
            empty_weight = self.empty_weight.clone()
            smooth_weights = self.smooth_weights.to(empty_weight.device)
            empty_weight[:-1] *= smooth_weights
            if hdhl:
                src_logits_filtered = src_logits[idx] # N*C
                target_classes_filtered = target_classes[idx]
                loss_ce = F.cross_entropy(src_logits_filtered, target_classes_filtered, empty_weight)
            else:
                loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, memory_part=None, hdhl=False):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx_new(self, indices, len_targets):
        # permute predictions following indices of current targets
        # print(indices, len_targets)
        batch_idx = torch.cat([torch.full_like(src[tgt >= len_targets[i]], i) for i, (src, tgt) in enumerate(indices)])
        src_idx = torch.cat([
            src[tgt >= len_targets[i]] for i, (src, tgt) in enumerate(indices)
        ])
        return batch_idx, src_idx

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, memory_part, hdhl):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, memory_part, hdhl)

    def get_mse_loss(self, med_tokens_to_old, old_med_tokens_to_old,  med_indices, num_layer):
        if med_indices.numel() == 0 or not med_indices.any():
            return med_tokens_to_old.sum() * 0.0
        mse_diffs = F.mse_loss(med_tokens_to_old[med_indices], old_med_tokens_to_old[med_indices], reduction='none')  # [N*B, C]

        med_indices = med_indices.transpose(0, 1)
        indices = torch.nonzero(med_indices)
        second_dim_indices = indices[:, 1]
        if (
            self.query_importance.numel() == 0
            or self.query_importance.shape[1] == 0
            or num_layer >= self.query_importance.shape[0]
        ):
            query_importance = torch.ones(
                second_dim_indices.shape[0],
                device=mse_diffs.device,
                dtype=mse_diffs.dtype,
            )
        else:
            layer_query_importance = self.query_importance[num_layer].to(mse_diffs.device)
            second_dim_indices = second_dim_indices.to(layer_query_importance.device)
            query_importance = layer_query_importance[second_dim_indices].to(mse_diffs.dtype)
        mse_diffs = mse_diffs * query_importance.unsqueeze(1)
        weighted_mse_loss = mse_diffs.mean()

        return weighted_mse_loss

    def safe_mse_loss(self, src, target):
        if src.numel() == 0 or target.numel() == 0:
            return src.sum() * 0.0
        return F.mse_loss(src, target)

    def cal_matrices(self, outputs, targets, outputs_without_aux, complete_psd_targets):
        # Retrieve the matching between the outputs of the last layer and the targets
        cost_matrices = []
        indices_layers = []
        for i, aux_outputs in enumerate(outputs["aux_outputs"]):
            indices, cost_matrix = self.matcher(aux_outputs, complete_psd_targets)
            cost_matrices.append(cost_matrix)
            indices_layers.append(indices)

        indices, cost_matrix = self.matcher(outputs_without_aux, complete_psd_targets)
        cost_matrices.append(cost_matrix)
        indices_layers.append(indices)
        return indices_layers, cost_matrices


    def forward(self, outputs, targets, psd_targets=None, old_targets=None, med_tokens=None, old_med_tokens=None,
                 flag=False, old_logits=None, old_aux_outputs=None, old_mask_features=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        if self.current_catagory_ids is not None:
            memory_part = [self._is_memory_sample(tgt) for tgt in targets]
        else:
            memory_part = None

        if psd_targets is None or old_targets is None:
            complete_psd_targets = targets
        else:
            complete_psd_targets = []

            for i, (p, t) in enumerate(zip(psd_targets, targets)):
                if memory_part[i]:
                    complete_psd_targets.append(
                        {'labels': t['labels'], 'masks': t['masks']}
                    )
                else:
                    complete_psd_targets.append(
                        {'labels': torch.cat([p['labels'], t['labels']]), 'masks': torch.cat([p['masks'], t['masks']])}
                    )
        indices_layers, cost_matrices = self.cal_matrices(outputs, targets, outputs_without_aux, complete_psd_targets)
        indices, cost_matrix = indices_layers[-1], cost_matrices[-1]

        if flag:
            self._ensure_importance_buffers(
                num_layers=len(cost_matrices),
                num_queries=cost_matrices[0][0].shape[0] if len(cost_matrices[0]) > 0 else outputs["pred_logits"].shape[1],
                device=outputs["pred_logits"].device,
            )

            for i, (cost_temp_matrix, indices_temp) in enumerate(zip(cost_matrices, indices_layers)):
                cost_query_list = []
                for j in range(len(cost_temp_matrix)):
                    current_matrix = cost_temp_matrix[j]
                    if current_matrix.shape[1] != 0:
                        column_means = current_matrix.mean(dim=0)
                        current_matrix = current_matrix / column_means

                        min_values, _ = current_matrix.topk(1, dim=1, largest=False)
                        min_values_sum = min_values.sum(dim=1)
                    else:
                        min_values_sum = torch.zeros(current_matrix.shape[0])

                    cost_query_list.append(min_values_sum)
                cost_query = torch.stack(cost_query_list)
                for cost in cost_query:
                    self.query_importance_accumulator[i] += cost.to(self.query_importance_accumulator[i].device)

            return {}

        if old_targets is not None:
            old_target_num = [len(t["labels"]) for t in old_targets]
            indices_to_old = \
                [ind_query[ind_target < old_target_num[i]] for i, (ind_query, ind_target) in enumerate(indices)]
            med_tokens_to_old = []
            old_med_tokens_to_old = []
            for per_med_tokens, per_old_med_tokens in zip(med_tokens, old_med_tokens):
                per_med_tokens_to_old = []
                per_old_med_tokens_to_old = []
                for i in range(per_med_tokens.shape[1]):
                    per_med_tokens_to_old.append(per_med_tokens[:, i][indices_to_old[i]])
                    per_old_med_tokens_to_old.append(per_old_med_tokens[:, i][indices_to_old[i]])
                med_tokens_to_old.append(torch.cat(per_med_tokens_to_old, dim=0))
                old_med_tokens_to_old.append(torch.cat(per_old_med_tokens_to_old, dim=0))

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in complete_psd_targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, complete_psd_targets, indices, num_masks, memory_part, old_logits is not None))
        if med_tokens is not None:
            losses.update({'loss_med_tokens': self.get_mse_loss(med_tokens[-1], old_med_tokens[-1],
                                                                 torch.ones(med_tokens[-1].shape[:2], device=med_tokens[-1].device).bool(), 9)})

        # Distill the logits
        if old_logits is not None:
            src_logits = outputs["pred_logits"].float()
            idx = self._get_src_permutation_idx(indices)
            negative_idx = torch.ones(src_logits.size(0), src_logits.size(1), device=src_logits.device)
            negative_idx[idx] = 0
            label_for_distill = negative_idx.bool()
            # Select logits of bg
            src_masked_logits = src_logits[label_for_distill]#[:, :-1]
            old_selected_logits_sm = old_logits[label_for_distill].softmax(dim=-1)
            old_selected_logits_sm = torch.cat([old_selected_logits_sm[:, :self.current_catagory_ids[0]],
                                                torch.zeros(old_selected_logits_sm.size(0), self.current_catagory_ids[-1] - self.current_catagory_ids[0]+1, device=old_selected_logits_sm.device),
                                                old_selected_logits_sm[:, -1:]
                                                ], dim=1)
            loss_ce_distill = F.kl_div(torch.log_softmax(src_masked_logits, dim=-1), old_selected_logits_sm, reduction='batchmean')

            losses.update({"loss_ce_distill": loss_ce_distill})

        if old_mask_features is not None:
            mask_features = outputs["mask_features"]
            union_target = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()[0].to(mask_features.device)
            union_target = union_target.max(1, keepdim=True)[0] if union_target.size(1) >= 1 else torch.zeros(union_target.size(0), 1, *union_target.shape[2:], device=union_target.device)
            union_target = F.interpolate(union_target.float(), size=mask_features.shape[-2:], mode='bilinear', align_corners=False)
            loss_feats = F.mse_loss(mask_features * (1 - union_target), old_mask_features * (1 - union_target), reduction='mean')
            losses.update({"loss_feats": loss_feats})

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # indices, C1 = self.matcher(aux_outputs, complete_psd_targets)
                if old_logits is not None and i == 0:
                    continue
                indices = indices_layers[i]
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, complete_psd_targets, indices, num_masks, memory_part,
                                           old_logits is not None)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if old_aux_outputs is not None:
                    src_logits = aux_outputs["pred_logits"].float()
                    idx = self._get_src_permutation_idx(indices)
                    negative_idx = torch.ones(src_logits.size(0), src_logits.size(1), device=src_logits.device)
                    negative_idx[idx] = 0
                    label_for_distill = negative_idx.bool()
                    src_masked_logits = src_logits[label_for_distill]#[:, :-1]
                    old_selected_logits_sm = old_aux_outputs[i]["pred_logits"][label_for_distill].softmax(dim=-1)
                    old_selected_logits_sm = torch.cat([old_selected_logits_sm[:, :self.current_catagory_ids[0]],
                                                        torch.zeros(old_selected_logits_sm.size(0), self.current_catagory_ids[-1] - self.current_catagory_ids[0]+1, device=old_selected_logits_sm.device),
                                                        old_selected_logits_sm[:, -1:]
                                                        ], dim=1)
                    # loss_ce_distill = F.binary_cross_entropy_with_logits(src_masked_logits, old_selected_logits_sm)
                    loss_ce_distill = F.kl_div(torch.log_softmax(src_masked_logits, dim=-1), old_selected_logits_sm, reduction='batchmean')
                    losses.update(
                        {f'loss_ce_distill_{i}': loss_ce_distill}
                    )
                if med_tokens is not None:
                    # losses.update(
                    #     {'loss_med_tokens_{}'.format(str(i)):
                    #          self.get_mse_loss(med_tokens_to_old[i], old_med_tokens_to_old[i], importance_cost[i],
                    #                            med_indices[i], i)}
                    # )
                    losses.update(
                        {'loss_med_tokens_{}'.format(str(i)):
                             self.safe_mse_loss(med_tokens_to_old[i], old_med_tokens_to_old[i]) / 3 * 5}
                    )

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
