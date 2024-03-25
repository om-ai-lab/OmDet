from detectron2.config import CfgNode as CN


def add_backbone_config(cfg):
    add_convnext_config(cfg)
    add_swint_config(cfg)


def add_convnext_config(cfg):
    # extra configs for convnext
    cfg.MODEL.CONVNEXT = CN()
    cfg.MODEL.CONVNEXT.SIZE = "T"
    cfg.MODEL.CONVNEXT.DEPTHS= [3, 3, 9, 3]
    cfg.MODEL.CONVNEXT.DIMS= [96, 192, 384, 768]
    cfg.MODEL.CONVNEXT.DROP_PATH_RATE= 0.2
    cfg.MODEL.CONVNEXT.LAYER_SCALE_INIT_VALUE= 1e-6
    cfg.MODEL.CONVNEXT.OUT_FEATURES= [0, 1, 2, 3]
    cfg.SOLVER.WEIGHT_DECAY_RATE= 0.95


def add_swint_config(cfg):
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'T'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3) # FPN stride 8 - 32


