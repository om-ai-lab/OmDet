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
__all__ = ['OmDetV2Turbo']

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from ..utils.cache import LRUCache


@META_ARCH_REGISTRY.register()
class OmDetV2Turbo(nn.Module):

    @configurable
    def __init__(self, cfg):
        super(OmDetV2Turbo, self).__init__()
        self.cfg = cfg
        self.logger = setup_logger(name=__name__)

        self.backbone = build_backbone(cfg)
        self.decoder = build_decoder_model(cfg)
        self.neck = build_encoder_model(cfg)
        self.loss_head = build_detr_head(cfg)
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

        # Build language Encoder
        self.language_backbone = build_language_backbone(cfg)
        self.language_cache_label = LRUCache(100)
        self.language_cache_prompt = LRUCache(100)


    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        return {
            'cfg': cfg
        }

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)
        ann_types = [x["ann_type"] if "ann_type" in x else "box" for x in batched_inputs]
        return images, images_whwh, ann_types

    def gen_output(self, box_cls, box_pred, batched_inputs, images, score_thresh, nms_thresh, do_postprocess,
                   max_num_det=None):
        results = self.inference(box_cls, box_pred, images.image_sizes, score_thresh, nms_thresh, max_num_det)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            results = processed_results
        return results

    def inference(self, box_cls, box_pred, image_sizes, score_thresh=None, nms_thresh=None, max_num_det=None):
        assert len(box_cls) == len(image_sizes)
        if score_thresh is None:
            score_thresh = self.conf_test_th

        if nms_thresh is None:
            nms_thresh = self.nms_test_th

        num_classes = box_cls.shape[2]
        scores, labels = self.compute_score(box_cls)
        results = []
        if self.loss_type in {"FOCAL", "BCE"}:
            for i, (scores_img, box_per_img, image_size) in enumerate(zip(scores, box_pred, image_sizes
                                                                          )):
                results.append(self.inference_single_image(box_per_img, scores_img, labels, image_size, num_classes,
                                                           score_thresh=score_thresh,
                                                           nms_thresh=nms_thresh,
                                                           max_num_det=max_num_det))
        else:
            for i, (scores_img, label_img, box_per_img, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                results.append(
                    self.inference_single_image(box_per_img, scores_img, label_img, image_size, num_classes,
                                                score_thresh=score_thresh,
                                                nms_thresh=nms_thresh,
                                                max_num_det=max_num_det))

        return results

    def inference_single_image(self, boxes, scores, labels,
                               image_size: Tuple[int, int],
                               num_classes: int,
                               score_thresh: float,
                               nms_thresh: float,
                               max_num_det: int = None):
        """
        Call `fast_rcnn_inference_single_image` for all images.
        Args:
            boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
                boxes for each image. Element i has shape (Ri, K * 4) if doing
                class-specific regression, or (Ri, 4) if doing class-agnostic
                regression, where Ri is the number of predicted objects for image i.
                This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
            scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
            image_size (list[tuple]): A list of (width, height) tuples for each image in the batch.
            score_thresh (float): Only return detections with a confidence score exceeding this
                threshold.
            nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        Returns:
            instances: (list[Instances]): A list of N instances, one for each image in the batch,
                that stores the topk most confidence detections.
            kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
                the corresponding boxes/scores index in [0, Ri) from the input, for image i.
        """
        # scores_per_image: num_proposal
        # labels_per_image: num_proposal
        # box_per_images: num_proposal x 4'
        if self.loss_type in {"FOCAL", "BCE"}:
            proposal_num = len(boxes) if max_num_det is None else max_num_det
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(proposal_num, sorted=False)
            labels_per_image = labels[topk_indices]
            box_pred_per_image = boxes.view(-1, 1, 4).repeat(1, num_classes, 1).view(-1, 4)
            box_pred_per_image = box_pred_per_image[topk_indices]
        else:
            box_pred_per_image = boxes
            scores_per_image = scores
            labels_per_image = labels

        # Score filtering
        box_pred_per_image = bbox_cxcywh_to_xyxy(box_pred_per_image) * torch.tensor(image_size).repeat(2).to(self.device)
        filter_mask = scores_per_image > score_thresh  # R x K
        score_keep = filter_mask.nonzero(as_tuple=False).view(-1)
        box_pred_per_image = box_pred_per_image[score_keep]
        scores_per_image = scores_per_image[score_keep]
        labels_per_image = labels_per_image[score_keep]

        # NMS
        scores_per_image.to(self.device)
        keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, nms_thresh)
        box_pred_per_image = box_pred_per_image[keep]
        scores_per_image = scores_per_image[keep]
        labels_per_image = labels_per_image[keep]

        # create an instance
        result = Instances(image_size)
        result.pred_boxes = Boxes(box_pred_per_image)
        result.pred_boxes.clip(image_size)
        result.scores = scores_per_image
        result.pred_classes = labels_per_image

        return result

    def compute_score(self, box_cls):
        """
        Args:
            box_cls: tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.

        Returns:
        """
        if self.loss_type in {"FOCAL", "BCE"}:
            num_classes = box_cls.shape[2]
            proposal_num = box_cls.shape[1]
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(num_classes, device=self.device). \
                unsqueeze(0).repeat(proposal_num, 1).flatten(0, 1)
        else:
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)
            # scores: batch_size x num_proposal

        return scores, labels

    def language_encode(self, batched_inputs, encode_type="task"):
        texts = batched_inputs

        if self.language_encoder_type == "clip":
            text_input = clip.tokenize(texts, truncate=True).to(self.device)

        return self.language_backbone(text_input, encode_type == "task")

    def get_cached_label_emb(self, labels):
        self.logger.info('processing labels embeddings for {}'.format(labels))
        not_cached_index = []
        not_cached_labels = []
        total_embs = []
        for idx, l in enumerate(labels):
            if self.language_cache_label.has(l):
                total_embs.append(self.language_cache_label.get(l))
            else:
                total_embs.append(None)
                not_cached_index.append(idx)
                not_cached_labels.append(l)

        self.logger.info('cached label emb num: {}, not cached num: {}'.format(len(total_embs) - len(not_cached_labels),
                                                                               len(not_cached_labels)))

        if not_cached_labels:
            embeddings = self.language_encode(not_cached_labels, encode_type="label")
            for idx, emb in enumerate(embeddings):
                idx_to_put = not_cached_index[idx]
                total_embs[idx_to_put] = emb
                self.language_cache_label.put(not_cached_labels[idx], emb)

        total_label_embs = torch.stack(total_embs).to(self.device)
        return total_label_embs

    def get_cached_prompt_emb(self, batched_tasks):
        self.logger.info('processing prompt embeddings for {}'.format(batched_tasks))
        not_cached_index = []
        not_cached_tasks = []
        total_task_features = []
        total_task_masks = []
        for idx, t in enumerate(batched_tasks):
            if self.language_cache_prompt.has(t):
                task_feature, task_mask = self.language_cache_prompt.get(t)
                total_task_features.append(task_feature)
                total_task_masks.append(task_mask)
            else:
                total_task_features.append(None)
                total_task_masks.append(None)
                not_cached_index.append(idx)
                not_cached_tasks.append(t)

        self.logger.info(
            'cached prompt emb num: {}, not cached num: {}'.format(len(total_task_features) - len(not_cached_tasks),
                                                                  len(not_cached_tasks)))

        if not_cached_tasks:
            embeddings, task_masks = self.language_encode(not_cached_tasks, encode_type="task")

            for idx in range(embeddings.shape[1]):
                emb = embeddings[:, [idx], :]
                idx_to_put = not_cached_index[idx]
                cur_mask = torch.unsqueeze(task_masks[idx], dim=0).to(self.device)
                total_task_features[idx_to_put] = emb
                total_task_masks[idx_to_put] = cur_mask
                self.language_cache_prompt.put(not_cached_tasks[idx], (emb, cur_mask))

        total_prompt_features = torch.cat(total_task_features, dim=1)
        total_prompt_masks = torch.cat(total_task_masks, dim=0).to(self.device)

        return total_prompt_features, total_prompt_masks

    def get_language_embedding(self, batched_inputs):
        batched_labels = [a["label_set"] for a in batched_inputs]
        batched_tasks = [a['tasks'] for a in batched_inputs]

        max_label_size = max([len(a) for a in batched_labels])
        label_features = []
        for i, s_labels in enumerate(batched_labels):
            pad_size = max_label_size - len(s_labels)

            label_emb = self.get_cached_label_emb(s_labels)
            label_features.append(F.pad(label_emb, (0, 0, 0, pad_size)).unsqueeze(1).to(self.device))

        label_features = torch.cat(label_features, dim=1)  # num_label x batch_size x dim_size

        # Task Features
        # prompt_features: max_task_len x batch_size x dim_size
        # prompt_mask: batch_size x max_task_len
        # batched_tasks = ['detect a person', 'detect dog and cat']
        prompt_features, prompt_mask = self.get_cached_prompt_emb(batched_tasks)

        return label_features, prompt_features, prompt_mask

    def forward(self, batched_inputs, do_postprocess=True, score_thresh=0.0, nms_thresh=1.0, debug=False):
        images, images_whwh, ann_types = self.preprocess_image(batched_inputs)

        # Backbone
        body_feats = self.backbone(images.tensor)

        if type(body_feats) is dict:
            body_feats = [body_feats[i] for i in body_feats.keys()]

        encoder_feats = self.neck(body_feats)

        if not self.training:
            # create label and prompt embeddings
            label_feats, prompt_feats, prompt_mask = self.get_language_embedding(batched_inputs)
            decoder_feats = self.decoder(encoder_feats, label_feats, prompt_feats, prompt_mask)
            box_pred, box_cls, _ = self.loss_head(decoder_feats)

            results = self.gen_output(box_cls, box_pred, batched_inputs, images,
                                      score_thresh, nms_thresh, do_postprocess,
                                      max_num_det=self.num_proposals)

        return results

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )