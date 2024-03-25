from torch import nn
from .build_components import DETR_HEAD_REGISTRY


__all__ = ['DINOHead']
@DETR_HEAD_REGISTRY.register()
class DINOHead(nn.Module):
    def __init__(self, device="cuda"):
        super(DINOHead, self).__init__()

    def forward(self, out_transformer, inputs=None):
        (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits,
         dn_meta) = out_transformer

        return (dec_out_bboxes[-1], dec_out_logits[-1], None)

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        return {
            "device": cfg.MODEL.DEVICE
        }
