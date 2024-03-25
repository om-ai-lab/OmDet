from detectron2.config import CfgNode as CN
from omdet.modeling.backbone.config import add_backbone_config


def add_omdet_v2_turbo_config(cfg):
    """
    Add config for Modulated OmDet Turn.
    """
    cfg.MODEL.HEAD = "DINOHead"
    cfg.MODEL.LOSS = "DINOLoss"
    cfg.MODEL.TRANSFORMER_ENCODER = "ELAEncoder"
    cfg.MODEL.TRANSFORMER_DECODER = "ELADecoder"

    cfg.MODEL.LANGUAGE_BACKBONE = CN()
    cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE = "clip"
    cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM = 512

    # Task Head
    cfg.MODEL.ELAEncoder = CN()
    cfg.MODEL.ELAEncoder.in_channels = [192, 384, 768]
    cfg.MODEL.ELAEncoder.feat_strides = [8, 16, 32]
    cfg.MODEL.ELAEncoder.hidden_dim = 256
    cfg.MODEL.ELAEncoder.use_encoder_idx = [2]
    cfg.MODEL.ELAEncoder.num_encoder_layers = 1
    cfg.MODEL.ELAEncoder.encoder_layer = 'TransformerLayer'
    cfg.MODEL.ELAEncoder.pe_temperature = 10000
    cfg.MODEL.ELAEncoder.expansion = 1.0
    cfg.MODEL.ELAEncoder.depth_mult = 1.0
    cfg.MODEL.ELAEncoder.act = 'silu'
    cfg.MODEL.ELAEncoder.eval_size = None
    cfg.MODEL.ELAEncoder.dim_feedforward=1024

    cfg.MODEL.ELADecoder = CN()
    cfg.MODEL.ELADecoder.hidden_dim = 256
    cfg.MODEL.ELADecoder.num_queries = 300
    cfg.MODEL.ELADecoder.position_embed_type = 'sine'
    cfg.MODEL.ELADecoder.backbone_feat_channels = [256, 256, 256]
    cfg.MODEL.ELADecoder.feat_strides = [8, 16, 32]
    cfg.MODEL.ELADecoder.num_levels = 3
    cfg.MODEL.ELADecoder.num_decoder_points = 4
    cfg.MODEL.ELADecoder.nhead = 8
    cfg.MODEL.ELADecoder.num_decoder_layers = 3
    cfg.MODEL.ELADecoder.dim_feedforward = 1024
    cfg.MODEL.ELADecoder.dropout = 0.0
    cfg.MODEL.ELADecoder.activation = 'relu'
    cfg.MODEL.ELADecoder.num_denoising = 100
    cfg.MODEL.ELADecoder.label_noise_ratio = 0.5
    cfg.MODEL.ELADecoder.box_noise_scale = 1.0
    cfg.MODEL.ELADecoder.learnt_init_query = True
    cfg.MODEL.ELADecoder.eval_size = None
    cfg.MODEL.ELADecoder.eval_idx = -1
    cfg.MODEL.ELADecoder.eps = 1e-2
    cfg.MODEL.ELADecoder.cls_type = 'cosine'

    cfg.MODEL.FUSE_TYPE = None

    cfg.INPUT.RANDOM_CROP = None
    cfg.INPUT.RANDOM_CONTRAST = None
    cfg.INPUT.RANDOM_BRIGHTNESS = None
    cfg.INPUT.RANDOM_SATURATION = None

    cfg.MODEL.DEPLOY_MODE = False

    add_backbone_config(cfg)