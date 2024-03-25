import torch
import torch.nn as nn
import torch.nn.functional as F
from .torch_utils import BaseConv, linear_init_
from .block import RepC3
from .detr_torch import TransformerEncoder
from .build_components import TRANSFORMER_ENCODER_REGISTRY

__all__ = ['ELAEncoder']


class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src = self.self_attn(q, k, value=src, attn_mask=src_mask)
        #print(src[1].shape, src[0].shape)
        src = src[0]
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


@TRANSFORMER_ENCODER_REGISTRY.register()
class ELAEncoder(nn.Module):
    # __shared__ = ['depth_mult', 'act', 'trt', 'eval_size']
    # __inject__ = ['encoder_layer']

    def __init__(self,
                 in_channels=[128, 256, 512],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 encoder_layer='TransformerLayer',
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 trt=False,
                 dim_feedforward=1024,
                 eval_size=None):
        super(ELAEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size

        self.encoder_layer = TransformerLayer(dim_feedforward=dim_feedforward)

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in self.in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channel, hidden_dim, kernel_size=(1, 1), bias=False),
                    nn.BatchNorm2d(
                        hidden_dim)))
        # encoder transformer
        self.encoder = nn.ModuleList([
            TransformerEncoder(self.encoder_layer, num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        # act = get_act_fn(
        #     act, trt=trt) if act is None or isinstance(act,
        #                                                (str, dict)) else act
        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for idx in range(len(self.in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                RepC3(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    e=1.0))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for idx in range(len(self.in_channels) - 1):
            self.downsample_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 3, stride=2, act=act))
            self.pan_blocks.append(
                RepC3(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    e=1.0))

    #     self._reset_parameters()
    #
    # def _reset_parameters(self):
    #     if self.eval_size:
    #         for idx in self.use_encoder_idx:
    #             stride = self.feat_strides[idx]
    #             pos_embed = self.build_2d_sincos_position_embedding(
    #                 self.eval_size[1] // stride, self.eval_size[0] // stride,
    #                 self.hidden_dim, self.pe_temperature)
    #             setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w,
                                           h,
                                           embed_dim=256,
                                           temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @omega[None]
        out_h = grid_h.flatten()[..., None] @omega[None]

        return torch.cat(
            [
                torch.sin(out_w), torch.cos(out_w), torch.sin(out_h),
                torch.cos(out_h)
            ],
            dim=1)[None, :, :]
    @classmethod
    def from_config(cls, cfg):
        enc_cfg = cfg.MODEL.ELAEncoder
        return {
            'in_channels': enc_cfg.in_channels,
            'feat_strides': enc_cfg.feat_strides,
            'hidden_dim': enc_cfg.hidden_dim,
            'use_encoder_idx': enc_cfg.use_encoder_idx,
            'num_encoder_layers': enc_cfg.num_encoder_layers,
            'encoder_layer': enc_cfg.encoder_layer,
            'pe_temperature': enc_cfg.pe_temperature,
            'expansion': enc_cfg.expansion,
            'depth_mult': enc_cfg.depth_mult,
            'act': enc_cfg.act,
            'eval_size': enc_cfg.eval_size,
            'dim_feedforward': enc_cfg.dim_feedforward
        }

    def forward(self, feats, for_mot=False):
        assert len(feats) == len(self.in_channels)
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(start_dim=2).transpose(1, 2)
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.transpose(1, 2).reshape((-1, self.hidden_dim, h, w))

        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(
                feat_heigh, scale_factor=2., mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.cat(
                    [upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.cat(
                [downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs
