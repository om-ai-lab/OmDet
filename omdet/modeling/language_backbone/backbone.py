import torch
from omdet.modeling import registry
from omdet.modeling.language_backbone.clip.models import clip as clip


@registry.LANGUAGE_BACKBONES.register("clip")
def build_clip_backbone(cfg):
    model, _ = clip.load("resources/ViT-B-16.pt", device=torch.device(cfg.MODEL.DEVICE), jit=False)
    model.visual = None # delete the vision part
    model.logit_scale = None
    return model


def build_language_backbone(cfg):
    print ("cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE", cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE)
    assert cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE in registry.LANGUAGE_BACKBONES, \
        "cfg.MODEL.LANGUAGE_BACKBONE.TYPE: {} is not registered in registry".format(
            cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        )
    return registry.LANGUAGE_BACKBONES[cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE](cfg)


if __name__ == "__main__":
    a = build_clip_backbone('')
    print(a)