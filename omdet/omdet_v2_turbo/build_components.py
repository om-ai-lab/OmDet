import torch
from detectron2.utils.logger import _log_api_usage
from detectron2.utils.registry import Registry

TRANSFORMER_ENCODER_REGISTRY = Registry("TRANSFORMER_ENCODER")  # noqa F401 isort:skip
TRANSFORMER_ENCODER_REGISTRY.__doc__ = """
"""

TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_DECODER")  # noqa F401 isort:skip
TRANSFORMER_DECODER_REGISTRY.__doc__ = """ """

DETR_HEAD_REGISTRY = Registry("DETR_HEAD")  # noqa F401 isort:skip
DETR_HEAD_REGISTRY.__doc__ = """ """


def build_encoder_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    encoder = cfg.MODEL.TRANSFORMER_ENCODER
    mode_class = TRANSFORMER_ENCODER_REGISTRY.get(encoder)
    model = mode_class(**mode_class.from_config(cfg))
    # model = TRANSFORMER_ENCODER_REGISTRY.get(encoder)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.transfor_encoder." + encoder)
    return model


def build_decoder_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    decoder = cfg.MODEL.TRANSFORMER_DECODER
    mode_class = TRANSFORMER_DECODER_REGISTRY.get(decoder)
    model = mode_class(**mode_class.from_config(cfg))
    # model = TRANSFORMER_DECODER_REGISTRY.get(decoder)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.transfor_encoder." + decoder)
    return model


def build_detr_head(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    head = cfg.MODEL.HEAD
    # model = DETR_HEAD_REGISTRY.get(head)(cfg)
    mode_class = DETR_HEAD_REGISTRY.get(head)
    model = mode_class(**mode_class.from_config(cfg))
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.transfor_encoder." + head)
    return model