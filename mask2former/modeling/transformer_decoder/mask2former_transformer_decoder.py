# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional

import numpy as np
import torch
from numpy import shape
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     return_attn: bool = False):
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                         key=self.with_pos_embed(memory, pos),
                                         value=memory, attn_mask=memory_mask,
                                         key_padding_mask=memory_key_padding_mask,
                                         need_weights=return_attn,
                                         average_attn_weights=False)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        if return_attn:
            return tgt, attn
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    return_attn: bool = False):
        tgt2 = self.norm(tgt)
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                         key=self.with_pos_embed(memory, pos),
                                         value=memory, attn_mask=memory_mask,
                                         key_padding_mask=memory_key_padding_mask,
                                         need_weights=return_attn,
                                         average_attn_weights=False)
        tgt = tgt + self.dropout(tgt2)

        if return_attn:
            return tgt, attn
        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_attn: bool = False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos, return_attn)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos, return_attn)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _maybe_upgrade_future_proxy_checkpoint(self, state_dict, prefix):
        if self.use_voc_future_proxy:
            return

        base_weight_key = prefix + "class_embeds.0.weight"
        base_bias_key = prefix + "class_embeds.0.bias"
        if base_weight_key not in state_dict or base_bias_key not in state_dict:
            return

        loaded_base_weight = state_dict[base_weight_key]
        loaded_base_bias = state_dict[base_bias_key]
        expected_base_out = self.class_embeds[0].out_features
        if (
            loaded_base_weight.ndim != 2
            or loaded_base_bias.ndim != 1
            or loaded_base_weight.shape[0] != expected_base_out + 1
            or loaded_base_bias.shape[0] != expected_base_out + 1
            or self.base_cls >= loaded_base_weight.shape[0] - 1
        ):
            return

        logger = logging.getLogger(__name__)
        future_proxy_weight = loaded_base_weight[self.base_cls].clone()
        future_proxy_bias = loaded_base_bias[self.base_cls].clone()
        state_dict[base_weight_key] = torch.cat(
            [loaded_base_weight[:self.base_cls], loaded_base_weight[-1:].clone()], dim=0
        )
        state_dict[base_bias_key] = torch.cat(
            [loaded_base_bias[:self.base_cls], loaded_base_bias[-1:].clone()], dim=0
        )

        if self.use_voc_future_proxy_init:
            for head_idx in range(1, len(self.class_embeds)):
                head_out = self.class_embeds[head_idx].out_features
                head_weight_key = prefix + f"class_embeds.{head_idx}.weight"
                head_bias_key = prefix + f"class_embeds.{head_idx}.bias"
                if head_weight_key not in state_dict:
                    state_dict[head_weight_key] = future_proxy_weight.unsqueeze(0).repeat(head_out, 1)
                if head_bias_key not in state_dict:
                    state_dict[head_bias_key] = future_proxy_bias.repeat(head_out)
            logger.info(
                "Initialized incremental classifier head(s) from the step-1 VOC future proxy in %s",
                base_weight_key,
            )
        else:
            logger.info(
                "Dropped the step-1 VOC future proxy row while loading %s into a later task model",
                base_weight_key,
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

        self._maybe_upgrade_future_proxy_checkpoint(state_dict, prefix)

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        n_cls_in_tasks: list,
        inc_cls: int, 
        rank: int,
        use_voc_future_proxy: bool,
        use_voc_future_proxy_init: bool,
        importance_grad_routing: bool,
        importance_grad_min: float,
        use_semantic_pixel_cross_attention: bool,
        detach_pixel_features: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.factorize_start_layer = max(0, self.num_layers - 3)
        self.use_voc_future_proxy = use_voc_future_proxy
        self.use_voc_future_proxy_init = use_voc_future_proxy_init
        self.importance_grad_routing = importance_grad_routing
        self.importance_grad_min = importance_grad_min
        if not 0.0 <= self.importance_grad_min <= 1.0:
            raise ValueError("CONT.IMPORTANCE_GRAD_MIN must be in [0, 1]")
        self.register_buffer(
            "routing_query_importance", torch.zeros(0, 0), persistent=False
        )
        self.use_semantic_pixel_cross_attention = use_semantic_pixel_cross_attention
        self.detach_pixel_features = detach_pixel_features
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.semantic_reader_layers = nn.ModuleList()
        self.semantic_pixel_cross_attention_layers = nn.ModuleList()
        self.num_queries = num_queries
        self._hook = None
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.semantic_reader_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # Preserve the released checkpoint layout and initialization sequence.
        # The final recipe keeps this embedding at a zero semantic offset.
        self.sem_query_feat = nn.Embedding(num_queries, hidden_dim)
        nn.init.zeros_(self.sem_query_feat.weight)
        if self.use_semantic_pixel_cross_attention:
            for _ in range(self.num_layers - self.factorize_start_layer):
                self.semantic_pixel_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.base_cls = n_cls_in_tasks[0]
        self.inc_cls = inc_cls
        if len(n_cls_in_tasks) > 1:
            self.start_index = self.base_cls + self.inc_cls * (len(n_cls_in_tasks) - 2)
            self.end_index = self.base_cls + self.inc_cls * (len(n_cls_in_tasks) - 1)
        else:
            self.start_index = self.base_cls
            self.end_index = self.base_cls

        self.step = len(n_cls_in_tasks)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        # output FFNs
        if self.mask_classification:
            # self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.class_embeds = nn.ModuleList()
            self.adapt_embeds = nn.ModuleList()
            # self.adapt_embeds = nn.ParameterList()
            self.rank = 16
            for i, n_cls in enumerate(n_cls_in_tasks):

                if i == 0:
                    self.class_embeds.append(
                        nn.Linear(hidden_dim, n_cls + (2 if self.use_voc_future_proxy else 1))
                    )
                else:
                    self.class_embeds.append(nn.Linear(hidden_dim, n_cls))
                    # self.adapt_embeds.append(nn.Parameter(torch.zeros(n_cls, hidden_dim), requires_grad=(i==self.step - 1)))
                    for cls in range(n_cls):
                        self.adapt_embeds.append(
                            nn.Sequential(
                                nn.Linear(hidden_dim, self.rank), 
                                nn.ReLU(), 
                                nn.Linear(self.rank, hidden_dim)
                            )
                        )
                if i == len(n_cls_in_tasks) - 2:
                    # freeze the parameters of the previous incremental adapters
                    for param in self.adapt_embeds.parameters():
                        param.requires_grad = False
        if self.step > 1:
            for param in self.query_feat.parameters():
                param.requires_grad = False
            for param in self.sem_query_feat.parameters():
                param.requires_grad = False
            for param in self.query_embed.parameters():
                param.requires_grad = False

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def apply_semantic_qcr(self, sem_output, outputs_class):
        outputs_class_arg = torch.argmax(outputs_class, dim=-1) # [B, Q]
        outputs_incre_class_mask = torch.logical_and(
            self.base_cls <= outputs_class_arg, outputs_class_arg < self.end_index
        )
        selected_adapt_embeds = torch.zeros_like(sem_output).transpose(0, 1)
        adapted_single_outputs = []
        for cls_arg, single_output in zip(
            outputs_class_arg[outputs_incre_class_mask],
            sem_output.transpose(0, 1)[outputs_incre_class_mask],
        ):
            adapted_single_outputs.append(self.adapt_embeds[cls_arg - self.base_cls](single_output))
        if len(adapted_single_outputs) > 0:
            selected_adapt_embeds[outputs_incre_class_mask] = torch.stack(
                adapted_single_outputs, dim=0
            ).float()
        selected_adapt_embeds = selected_adapt_embeds.transpose(0, 1)
        return sem_output + selected_adapt_embeds

    def set_routing_query_importance(self, query_importance):
        if query_importance is None or query_importance.numel() == 0:
            self.routing_query_importance = torch.zeros(
                0, 0, device=self.query_feat.weight.device
            )
            return
        self.routing_query_importance = query_importance.detach().to(
            self.query_feat.weight.device
        ).clone()

    def _route_semantic_gradient(self, struct_output, importance_layer=None):
        """Keep shared proposal features unchanged while routing reader gradients."""
        if (
            not self.importance_grad_routing
            or self.step <= 1
            or self.routing_query_importance.numel() == 0
        ):
            return struct_output

        # CoMBO defines IKD importance on the final query representation.  The
        # same task-level vector controls all late reader-to-structure edges.
        importance = self.routing_query_importance[-1]
        if importance.numel() != struct_output.shape[0]:
            raise ValueError(
                "Query importance and decoder query counts differ: "
                f"{importance.numel()} vs {struct_output.shape[0]}"
            )

        importance = importance.to(device=struct_output.device, dtype=struct_output.dtype)
        rank = (importance[:, None] > importance[None, :]).sum(dim=1)
        rank = rank.to(struct_output.dtype) / max(importance.numel() - 1, 1)
        gate = self.importance_grad_min + (1.0 - self.importance_grad_min) * (1.0 - rank)
        gate = gate[:, None, None]
        detached = struct_output.detach()
        return detached + gate * (struct_output - detached)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        is_voc_dataset = bool(cfg.DATASETS.TRAIN) and "pascal" in cfg.DATASETS.TRAIN[0]
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        tot_cls = cfg.CONT.TOT_CLS
        base_cls = cfg.CONT.BASE_CLS
        inc_cls = cfg.CONT.INC_CLS
        task = cfg.CONT.TASK

        num_tasks = 1 + (tot_cls - base_cls) // inc_cls
        n_cls_in_tasks = [base_cls] + [inc_cls] * (num_tasks - 1)
        voc_future_proxy_enabled = is_voc_dataset and (
            cfg.CONT.VOC_FUTURE_PROXY or cfg.CONT.VOC_FUTURE_PROXY_INIT
        )
        if voc_future_proxy_enabled and task == 1:
            # Step-1 VOC proxy training should learn exactly one future proxy
            # without exposing later-task classifier heads to the step-1 softmax.
            n_cls_in_tasks = [base_cls]

        ret["inc_cls"] = inc_cls
        ret["n_cls_in_tasks"] = n_cls_in_tasks[:task]
        ret["rank"] = cfg.CONT.RANK
        ret["use_voc_future_proxy"] = bool(voc_future_proxy_enabled and cfg.CONT.TASK == 1)
        ret["use_voc_future_proxy_init"] = bool(cfg.CONT.VOC_FUTURE_PROXY_INIT and cfg.CONT.TASK > 1)
        ret["importance_grad_routing"] = bool(cfg.CONT.IMPORTANCE_GRAD_ROUTING)
        ret["importance_grad_min"] = float(cfg.CONT.IMPORTANCE_GRAD_MIN)
        ret["use_semantic_pixel_cross_attention"] = bool(
            cfg.CONT.SEMANTIC_PIXEL_CROSS_ATTENTION
        )
        ret["detach_pixel_features"] = bool(cfg.CONT.DETACH_PIXEL_FEATURES)

        return ret

    def forward(self, x, mask_features, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        struct_output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        sem_output = None
        med_queries = []

        # if self.step > 1:
        #     adapt_embeds = torch.cat([embed for embed in self.adapt_embeds], dim=0)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            struct_output, mask_features, attn_mask_target_size=size_list[0], sem_output=None
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention first
            struct_output = self.transformer_cross_attention_layers[i](
                struct_output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            struct_output = self.transformer_self_attention_layers[i](
                struct_output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            struct_output = self.transformer_ffn_layers[i](
                struct_output
            )

            if i >= self.factorize_start_layer:
                if sem_output is None:
                    base_state = self._route_semantic_gradient(struct_output, i + 1)
                    role_embed = self.sem_query_feat.weight.unsqueeze(1).expand(-1, bs, -1)
                    sem_output = base_state + 0.0 * role_embed
                reader_memory = self._route_semantic_gradient(struct_output, i + 1)
                if i == self.num_layers - 1:
                    # Keep the original attention execution path for numerical
                    # parity with the final model; the weights are unused.
                    sem_output, _ = self.semantic_reader_layers[i](
                        sem_output, reader_memory,
                        memory_mask=None,
                        memory_key_padding_mask=None,
                        pos=query_embed, query_pos=query_embed,
                        return_attn=True,
                    )
                else:
                    sem_output = self.semantic_reader_layers[i](
                        sem_output, reader_memory,
                        memory_mask=None,
                        memory_key_padding_mask=None,
                        pos=query_embed, query_pos=query_embed
                    )

                if self.use_semantic_pixel_cross_attention:
                    pixel_layer_idx = i - self.factorize_start_layer
                    pixel_memory = (
                        src[level_index].detach()
                        if self.detach_pixel_features
                        else src[level_index]
                    )
                    sem_output = self.semantic_pixel_cross_attention_layers[pixel_layer_idx](
                        sem_output,
                        pixel_memory,
                        memory_mask=attn_mask,
                        memory_key_padding_mask=None,
                        pos=pos[level_index],
                        query_pos=query_embed,
                    )

            ori_output = struct_output
            med_queries.append(ori_output)
            pred_sem_output = sem_output if i >= self.factorize_start_layer else None
            pred_sem_output = self.apply_semantic_qcr(pred_sem_output, outputs_class) if self.step > 1 and i >= self.num_layers - 1 else pred_sem_output
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                struct_output, mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                sem_output=pred_sem_output,
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
            'med_tokens': med_queries,
            'mask_features': mask_features,
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, sem_output=None):
        decoder_output = self.decoder_norm(output).transpose(0, 1)
        sem_decoder_output = self.decoder_norm(sem_output if sem_output is not None else output)
        sem_decoder_output = sem_decoder_output.transpose(0, 1)
        # outputs_class = self.class_embed(decoder_output)
        # outputs_class = torch.cat([cls_embed(decoder_output) for cls_embed in self.class_embeds], dim=2)
        outputs_class_list = [cls_embed(sem_decoder_output) for cls_embed in self.class_embeds]
        outputs_class_list.append(outputs_class_list[0][:, :, -1:])
        outputs_class_list[0] = outputs_class_list[0][:, :, :-1]
        outputs_class = torch.cat(outputs_class_list, dim=2)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
