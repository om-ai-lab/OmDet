import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_

__all__ = 'multi_scale_deformable_attn_pytorch', 'inverse_sigmoid'


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    return float(-np.log((1 - prior_prob) / prior_prob))  # return bias_init


def linear_init_(module):
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, 'bias') and module.bias is not None:
        uniform_(module.bias, -bound, bound)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def multi_scale_deformable_attn_pytorch(value: torch.Tensor, value_spatial_shapes: torch.Tensor,
                                        sampling_locations: torch.Tensor,
                                        attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Multi-scale deformable attention.
    https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = (value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_))
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(value_l_,
                                          sampling_grid_l_,
                                          mode='bilinear',
                                          padding_mode='zeros',
                                          align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, num_queries,
                                                                  num_levels * num_points)
    output = ((torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(
        bs, num_heads * embed_dims, num_queries))
    return output.transpose(1, 2).contiguous()


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

def cls_score(cls_type, cls_feature, class_proj, logit_scale):
    if cls_type == 'cosine':
        class_logits = _b_cosine(cls_feature, class_proj, logit_scale)  # 4 100 256 4 256 20
    elif cls_type == 'dot':
        class_logits = torch.bmm(cls_feature, class_proj)  # 4 100 20
    else:
        raise Exception("Unknown cls type {}".format(cls_type))
    return class_logits

def _norm(f, dim=-1):
    return f / f.norm(dim=dim, keepdim=True).clamp_min(1e-12)


def _b_cosine(a, b, logit_scale):
    """
    a: B x K x H
    b: B x H x K
    """
    a = _norm(a, dim=2)
    b = _norm(b, dim=1)
    # Calculating the Loss
    logit_scale = logit_scale.exp()
    logits_per_image = logit_scale * torch.bmm(a, b)
    return logits_per_image

###########################
def bbox_cxcywh_to_xyxy(x):
    cxcy, wh = torch.split(x, 2, dim=-1)
    return torch.cat([cxcy - 0.5 * wh, cxcy + 0.5 * wh], dim=-1)

def bbox_xyxy2cxcywh(x):
    x0, y0, x1, y1 = torch.split(x, 1, dim=-1)
    return torch.cat([(x1+x0)/2, (y1+y0)/2, x1-x0, y1-y0], dim=-1)

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class BaseConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act="silu"):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(
            out_channels,
            # epsilon=1e-3,  # for amp(fp16), set in ppdet/engine/trainer.py
            # momentum=0.97,
            # weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            # bias_attr=ParamAttr(regularizer=L2Decay(0.0))
            )

        if act == 'silu':
            self.act = SiLU()
        elif act == 'gelu':
            self.act = nn.GELU()
    #     self._init_weights()
    #
    # def _init_weights(self):
    #     conv_init_(self.conv)

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.training:
            y = self.act(x)
        else:
            if isinstance(self.act, nn.SiLU):
                self.act = SiLU()
            y = self.act(x)
        return y

import random
import torchvision

class BatchResize():
    def __init__(self, mode="training"):
        self.mode = mode
        if mode == "training":
            self.size = int(random.choice(np.arange(480, 801, step=32)))
        else:
            self.size = 640
        self.resize = torchvision.transforms.Resize((self.size, self.size))

    def __call__(self, batch_inputs):
        for i, b in enumerate(batch_inputs):
            h, w = batch_inputs[i]["image"].shape[1:]
            batch_inputs[i]["image"] = self.resize(batch_inputs[i]["image"])
            new_h, new_w = (self.size, self.size)
            if self.mode:
                batch_inputs[i]["instances"].gt_boxes.tensor *= torch.tensor([new_w/w, new_h/h]).repeat(1, 2)
                batch_inputs[i]["instances"]._image_size = (new_h, new_w)

        return batch_inputs


def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0):
    """
    targets: [targets] that contains labels, bboxes, etc
    num_classes: the size of labels
    num_queries: 300
    class_embed: num_class x batch_size x label_dim OR num_class x batch_size (in the old case)
    """
    if num_denoising <= 0:
        return None, None, None, None
    # number of gt_bboxes in each batch sample
    num_gts = [len(t["labels"]) for t in targets]
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num  # the number of denoising group given num_denoising
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(targets)
    input_query_class = torch.full((bs, max_gt_num), num_classes, dtype=torch.int32)  # batch_size x max_gt_num (initialized with num_class)
    input_query_bbox = torch.zeros((bs, max_gt_num, 4))  # batch_size x max_gt_num x 4
    pad_gt_mask = torch.zeros((bs, max_gt_num))
    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["labels"].squeeze(-1)
            input_query_bbox[i, :num_gt] = targets[i]["boxes"]
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])  # batch_size x (max_gt_num*2*num_group)
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1])  # bs x max_gt_num*2 x 1
    negative_gt_mask[:, max_gt_num:] = 1  # set the second half to be NEGATIVE
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])  # bs x max_gt_num*2*num_group x 1
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx,
                                   [n * num_group for n in num_gts]) # split by batch+soze
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        input_query_class = input_query_class.flatten()  # (batch_size*max_gt_num*2*num_group) * 1
        pad_gt_mask = pad_gt_mask.flatten()
        # half of bbox prob
        mask = torch.rand(input_query_class.shape) < (label_noise_ratio * 0.5)
        chosen_idx = torch.nonzero(mask * pad_gt_mask).squeeze(-1)
        # randomly put a new one here
        new_label = torch.randint_like(
            chosen_idx, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class.scatter_(0, chosen_idx, new_label)
        input_query_class = input_query_class.reshape(bs, num_denoising)
        pad_gt_mask = pad_gt_mask.reshape(bs, num_denoising)

    if box_noise_scale > 0:
        known_bbox = bbox_cxcywh_to_xyxy(input_query_bbox)

        diff = torch.tile(input_query_bbox[..., 2:] * 0.5,
                           [1, 1, 2]) * box_noise_scale

        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand(input_query_bbox.shape)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (
            1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = bbox_xyxy2cxcywh(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    fixed_class = class_embed.dim() == 2
    if fixed_class: # fixed class embedding. num_class * hidden_dim
        class_embed = torch.cat(
            [class_embed, torch.zeros([1, class_embed.shape[-1]], device=class_embed.device)])  # (num_class+1) * hidden_dim
    else:
        assert class_embed.dim() == 3
        # (num_class+1) x batch_size x hidden_dim
        class_embed = torch.cat(
            [class_embed, torch.zeros([1, class_embed.shape[-2], class_embed.shape[-1]], device=class_embed.device)])

    if fixed_class:
        input_query_class_index = input_query_class.view(input_query_class.shape[0], -1)\
            .long().flatten().reshape(-1,1).repeat(1, class_embed.shape[-1])
        input_query_class = torch.gather(class_embed.to(input_query_class_index.device),
                                         dim=0,
                                         index=input_query_class_index).reshape([bs, num_denoising, -1])
    else:
        temp = []
        input_query_class_index = input_query_class.view(input_query_class.shape[0], -1) \
            .long().flatten().reshape(-1, 1).repeat(1, class_embed.shape[-1]).reshape([bs, num_denoising, -1])
        for b_id in range(bs):
            t = torch.gather(class_embed[:, b_id].to(input_query_class_index.device),
                             dim=0, index=input_query_class_index[b_id])
            temp.append(t)
        input_query_class = torch.cat(temp, dim=0).reshape([bs, num_denoising, -1])

    tgt_size = num_denoising + num_queries
    attn_mask = torch.ones([tgt_size, tgt_size]) < 0
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True
    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), max_gt_num *
                      2 * (i + 1):num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), :max_gt_num *
                      i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), max_gt_num *
                      2 * (i + 1):num_denoising] = True
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), :max_gt_num *
                      2 * i] = True
    attn_mask = ~attn_mask
    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    return input_query_class, input_query_bbox, attn_mask, dn_meta







