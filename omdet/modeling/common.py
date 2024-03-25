import math
import torch
from torch import nn, Tensor
import copy
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class AbsPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        seq_len = x.size(0)
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)
        pos_emb = self.pe(position)
        x = x + pos_emb
        return self.dropout(x)


class ResMultiHeadAttention(nn.Module):
    def __init__(self, d_q, d_k, d_v, nhead, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_q, nhead, dropout=dropout, kdim=d_k, vdim=d_v)
        self.norm1 = nn.LayerNorm(d_q)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k=None, v=None, attn_mask=None):
        """
        """
        if k is None:
            k = q

        if v is None:
            v = q

        q1 = self.self_attn(query=q, key=k, value=v, attn_mask=attn_mask)[0]
        q = q + self.dropout(q1)
        q = self.norm1(q)
        return q


class DistilMLP(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1):
        super(DistilMLP, self).__init__()
        self.squash = nn.GELU()
        self.LayerNorm = nn.LayerNorm(input_size, eps=1e-12)
        self.intermediate = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(input_size, output_size)

    def forward(self, word_emb):
        word_emb = self.squash(word_emb)
        word_emb = self.LayerNorm(word_emb)
        word_emb = self.dropout(word_emb)
        word_emb = self.dense(word_emb)
        return word_emb


class ResidualLayer(nn.Module):
    """
    A residual connection followed by a layer norm.
    """
    def __init__(self, size, dropout):
        super(ResidualLayer, self).__init__()
        self.norm1 = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        "Apply residual connection to any sublayer with the same size."
        return self.norm1(x + self.dropout(y))


class ResidualMLP(nn.Module):
    def __init__(self, d_m, dropout, d_hidden=1024, activation='relu'):
        super(ResidualMLP, self).__init__()
        self.mlp = MLP(d_m, d_m, d_hidden, dropout, activation)
        self.res1 = ResidualLayer(d_m, dropout)

    def forward(self, x):
        mlp_out = self.mlp(x)
        x = self.res1(x, mlp_out)
        return x


class MLP(nn.Module):
    def __init__(self, d_input, d_output, d_hidden=1024, dropout=0.1, activation='relu'):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(d_input, d_hidden)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_output)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


def apply_deltas(deltas, boxes, bbox_weights, scale_clamp):
    """
    Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

    Args:
        deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
            deltas[i] represents k potentially different class-specific
            box transformations for the single box boxes[i].
        boxes (Tensor): boxes to transform, of shape (N, 4)
    """
    boxes = boxes.to(deltas.dtype)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = bbox_weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=scale_clamp)
    dh = torch.clamp(dh, max=scale_clamp)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

    return pred_boxes


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


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

def _cosine(a, b, logit_scale):
    """
    a: ?/1 x K x H
    b: ?/1 x H x 1
    """
    a = _norm(a, dim=2)
    b = _norm(b, dim=1)
    # Calculating the Loss
    logit_scale = logit_scale.exp()
    logits_per_image = logit_scale * torch.matmul(a, b)
    return logits_per_image