from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from typing import Tuple

import numpy as np
# import open_clip
from detectron2.structures import Boxes, ImageList, Instances
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import detector_postprocess
from detectron2.layers import batched_nms
from detectron2.modeling import build_backbone
from omdet.omdet_v2_turbo.build_components import build_encoder_model, build_decoder_model, build_detr_head
from detectron2.config import configurable
from omdet.modeling.language_backbone import build_language_backbone
from detectron2.utils.logger import setup_logger
from ..modeling.language_backbone.clip.models import clip as clip
from .torch_utils import bbox_cxcywh_to_xyxy
__all__ = ['OmDetV2TurboInfer']

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from ..utils.cache import LRUCache

from huggingface_hub import PyTorchModelHubMixin


@META_ARCH_REGISTRY.register()
class OmDetV2TurboInfer(nn.Module, PyTorchModelHubMixin):

    @configurable
    def __init__(self, cfg):
        super(OmDetV2TurboInfer, self).__init__()
        self.cfg = cfg
        self.logger = setup_logger(name=__name__)

        self.backbone = build_backbone(cfg)
        self.decoder = build_decoder_model(cfg)
        self.neck = build_encoder_model(cfg)
        self.device = cfg.MODEL.DEVICE

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.normalizer = normalizer

        self.size_divisibility = self.backbone.size_divisibility
        self.nms_test_th = 0.0
        self.conf_test_th = 0.0
        self.loss_type = 'FOCAL'
        self.use_language_cache = True
        self.language_encoder_type = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        self.num_proposals = cfg.MODEL.ELADecoder.num_queries


    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        return {
            'cfg': cfg
        }

    def forward(self, x, label_feats, task_feats, task_mask):

        body_feats = self.backbone(x)

        if type(body_feats) is dict:
            body_feats = [body_feats[i] for i in body_feats.keys()]
        encoder_feats = self.neck(body_feats)
        box_pred, box_cls, _, _, _ = self.decoder(encoder_feats, label_feats, task_feats, task_mask)

        return box_pred, box_cls
