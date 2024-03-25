import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from .build_components import TRANSFORMER_DECODER_REGISTRY
from .conv import Conv
from .torch_utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch, cls_score
from .torch_utils import bias_init_with_prob, linear_init_
from omdet.modeling.common import ResidualMLP


__all__ = ('TransformerEncoderLayer', 'TransformerLayer', 'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'AIFI',
           'ELADecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP')


@TRANSFORMER_DECODER_REGISTRY.register()
class ELADecoder(nn.Module):
    __shared__ = ['hidden_dim', 'eval_size']

    def __init__(
            self,
            label_dim=512,
            cls_type="cosine",
            ch=(256, 256, 256),
            hd=256,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            dropout=0.,
            act=nn.ReLU(),
            eval_idx=-1,
            # training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False,
            te_type='clip',
            fuse_type= "merged_attn",
            amp=False
    ):
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = 80
        self.num_queries = nq
        self.num_decoder_layers = ndl
        self.label_dim = label_dim
        self.cls_type = cls_type
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)
        self.amp = amp

        # backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)


        self.task_encoder = None
        self.fuse_type = fuse_type

        if fuse_type is not None:
            self.task_encoder = ResidualMLP(self.label_dim, dropout=dropout)

            if fuse_type == 'merged_attn' and self.label_dim != self.hidden_dim:
                self.task_project = nn.Linear(self.label_dim, self.hidden_dim)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp, self.fuse_type)
        self.decoder = DeformableTransformerDecoderV2(hd, decoder_layer, ndl, eval_idx, cls_type=self.cls_type)

        # denoising part
        #self.denoising_class_embed = nn.Embedding(self.nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.denoising_embed_proj = nn.Linear(self.label_dim, self.hidden_dim)

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        #self.enc_score_head = nn.Linear(hd, self.nc)
        self.enc_score_head = nn.Linear(self.label_dim, self.hidden_dim)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(self.label_dim, self.hidden_dim) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    @classmethod
    def from_config(cls, cfg):
        model_cfg = cfg.MODEL.ELADecoder
        return {
            'label_dim': cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM,
            'hd': model_cfg.hidden_dim,
            'nq': model_cfg.num_queries,
            #'position_embed_type': model_cfg.position_embed_type,
            'ch': model_cfg.backbone_feat_channels,
            #'feat_strides': model_cfg.feat_strides,
            #'num_levels': model_cfg.num_levels,
            'ndp': model_cfg.num_decoder_points,
            'nh': model_cfg.nhead,
            'ndl': model_cfg.num_decoder_layers,
            'd_ffn': model_cfg.dim_feedforward,
            'dropout': model_cfg.dropout,
            #'act': model_cfg.activation,
            'nd': model_cfg.num_denoising,
            'label_noise_ratio': model_cfg.label_noise_ratio,
            'box_noise_scale': model_cfg.box_noise_scale,
            'learnt_init_query': model_cfg.learnt_init_query,
            #'eval_size': model_cfg.eval_size,
            'eval_idx': model_cfg.eval_idx,
            #'eps': model_cfg.eps,
            'cls_type': model_cfg.cls_type,
            'te_type': cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE,
            'fuse_type': cfg.MODEL.FUSE_TYPE,
            "amp": cfg.SOLVER.AMP.ENABLED
        }

    def forward(self, x, label_feats, task_feats, task_mask, batch=None):
        #from ultralytics.vit.utils.ops import get_cdn_group
        from .dn_ops import get_cdn_group

        # input projection and embedding
        feats, shapes = self._get_encoder_input(x)
        num_classes = label_feats.shape[0]

        new_batch = {}
        if self.training:
            #new_batch["gt_groups"] = [x["groups"] for x in batch]
            new_batch["cls"] = torch.cat([b["labels"] for b in batch], dim=0)
            new_batch["bboxes"] = torch.cat([b["boxes"] for b in batch], dim=0)
            new_batch["gt_groups"] = [b["groups"] for b in batch]
            batch_idx = torch.tensor([], dtype=torch.int32)
            for i, idx in enumerate(new_batch["gt_groups"]):
                x = torch.tensor([i] * idx, dtype=torch.int32)
                batch_idx = torch.cat((batch_idx, x), 0)

            new_batch["batch_idx"] = batch_idx.to("cuda")

        # prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(new_batch,
                          num_classes,
                          self.num_queries,
                          self.denoising_embed_proj(label_feats),
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training,
                          self.amp)

        bs = task_mask.shape[0]

        # compose attn_mask for vision_emb and task_emb fusion
        if self.fuse_type == 'merged_attn':
            if self.task_encoder is not None:
                task_feats = self.task_encoder(task_feats)

            if self.task_project is not None:
                task_feats = self.task_project(task_feats)

            src_key_mask = (task_mask == 0).detach()
            if self.training and attn_mask is not None:
                attn_mask_len = attn_mask.shape[0]

                fusion_size = attn_mask.shape[0]+task_feats.shape[0]
                new_attn_mask = torch.zeros([bs, fusion_size, fusion_size], dtype=torch.bool)
                new_attn_mask[:, :attn_mask_len, :attn_mask_len] = attn_mask.unsqueeze(0).expand(bs, -1, -1)

                new_attn_mask[:, attn_mask_len:, :dn_embed.shape[2]] = True
                new_attn_mask[:, :, attn_mask_len:] = src_key_mask.unsqueeze(1)
                new_attn_mask = new_attn_mask.repeat(self.nhead, 1, 1)
                attn_mask = new_attn_mask.to(attn_mask.device)  # [bs, dn+num_query+task_token_len, dn+num_query+task_token_len]
            else:
                attn_mask_len = self.num_queries
                fusion_size = attn_mask_len + task_feats.shape[0]
                new_attn_mask = torch.zeros([bs, fusion_size, fusion_size], dtype=torch.bool)
                new_attn_mask[:, :, attn_mask_len:] = src_key_mask.unsqueeze(1)
                new_attn_mask = new_attn_mask.repeat(self.nhead, 1, 1)
                attn_mask = new_attn_mask.to(task_mask.device)

        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes,label_feats, dn_embed, dn_bbox)

        # decoder
        dec_bboxes, dec_scores = self.decoder(embed,
                                              refer_bbox,
                                              feats,
                                              shapes,
                                              label_feats,
                                              task_feats,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        return x

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        anchors = []
        for i, (h, w) in enumerate(shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=dtype, device=device),
                                            torch.arange(end=w, dtype=dtype, device=device),
                                            indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        # get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, label_feats, dn_embed=None, dn_bbox=None):
        bs = len(feats)
        # prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(torch.where(valid_mask, feats, 0.0))  # bs, h*w, 256

        #enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)
        clas_proj = self.enc_score_head(label_feats).permute(1, 2, 0)  #
        enc_outputs_scores = cls_score(self.cls_type, features, clas_proj, self.logit_scale)

        # dynamic anchors + static content
        enc_outputs_bboxes = self.enc_bbox_head(features) + anchors  # (bs, h*w, 4)

        # query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # Unsigmoided
        refer_bbox = enc_outputs_bboxes[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # refer_bbox = torch.gather(enc_outputs_bboxes, 1, topk_ind.reshape(bs, self.num_queries).unsqueeze(-1).repeat(1, 1, 4))

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        if self.training:
            refer_bbox = refer_bbox.detach()
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        if self.learnt_init_query:
            embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            embeddings = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
            if self.training:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        # class and bbox head init
        #bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        #constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init_(cls_)
            #constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos=None):
        """Add position embeddings if given."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([torch.sin(out_w), torch.cos(out_w),
                             torch.sin(out_h), torch.cos(out_h)], axis=1)[None, :, :]


class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c, num_heads):
        """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Apply a transformer block to the input x and return the output."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    """Vision Transformer https://arxiv.org/abs/2010.11929."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initialize a Transformer module with position embedding and specified number of heads and layers."""
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Forward propagates the input through the bottleneck module."""
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class MLPBlock(nn.Module):

    def __init__(self, embedding_dim, mlp_dim, act=nn.GELU):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


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


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MSDeformAttn(nn.Module):
    """
    Original Multi-Scale Deformable Attention Module.
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        assert _d_per_head * n_heads == d_model, '`d_model` must be divisible by `n_heads`'

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        Args:
            query (torch.Tensor): [bs, query_length, C]
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (torch.Tensor): [bs, value_length, C]
            value_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {num_points}.')
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    """
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(), n_levels=4, n_points=4,
                 fuse_type='merged_attn'):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.fuse_type = fuse_type

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, embed, task_feats, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        origin_emb_len = embed.shape[1]

        # self attention
        q = k = self.with_pos_embed(embed, query_pos)

        # combine task_emb with q, k, v
        if self.fuse_type == 'merged_attn':
            task_feats = task_feats.transpose(0, 1)  # [bs, token_len, hidden]
            q = torch.cat((q, task_feats), dim=1)    # [bs, dn+num_query+token_len, hidden]
            k = torch.cat((k, task_feats), dim=1)    # [bs, dn+num_query+token_len, hidden]
            embed = torch.cat((embed, task_feats), dim=1)  # [bs, dn+num_query+token_len, hidden]

        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1),
                             attn_mask=attn_mask)[0].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # cut fused embedd to vision emb and task emb         todo  spilt here or split before
        task_feats = embed[:, origin_emb_len:, :].transpose(0, 1)  # [token_len, bs, hidden]
        embed = embed[:, :origin_emb_len, :]   # [bs, dn+num_query, hidden]

        # cross attention
        tgt = self.cross_attn(self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes,
                              padding_mask)
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # ffn
        embed = self.forward_ffn(embed)

        return embed, task_feats


class DeformableTransformerDecoderV2(nn.Module):
    """
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1, cls_type='cosine'):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.cls_type = cls_type
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)

    def forward(
            self,
            embed,  # decoder embeddings
            refer_bbox,  # anchor
            feats,  # image features
            shapes,  # feature shapes
            label_feats, #label features
            task_feats,
            bbox_head,
            score_head,
            pos_mlp,
            attn_mask=None,
            padding_mask=None):
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output, task_feats = layer(output, task_feats, refer_bbox, feats, shapes, padding_mask, attn_mask,
                           pos_mlp(refer_bbox))

            # refine bboxes, (bs, num_queries+num_denoising, 4)
            refined_bbox = torch.sigmoid(bbox_head[i](output) + inverse_sigmoid(refer_bbox))

            clas_proj = score_head[i](label_feats).permute(1, 2, 0)

            if self.training:
                dec_cls.append(cls_score(self.cls_type, output, clas_proj, self.logit_scale))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox_head[i](output) + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(cls_score(self.cls_type, output, clas_proj, self.logit_scale))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)








