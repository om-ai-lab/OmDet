import os
import torch
from typing import List, Union, Dict
from omdet.utils.tools import chunks
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer as Trainer
from omdet.utils.cache import LRUCache
from omdet.inference.base_engine import BaseEngine
from detectron2.utils.logger import setup_logger
from omdet.omdet_v2_turbo.config import add_omdet_v2_turbo_config


class DetEngine(BaseEngine):
    def __init__(self, model_dir='resources/', device='cpu', batch_size=10):
        self.model_dir = model_dir
        self._models = LRUCache(10)
        self.device = device
        self.batch_size = batch_size
        self.logger = setup_logger(name=__name__)

    def _init_cfg(self, cfg, model_id):
        cfg.MODEL.WEIGHTS = os.path.join(self.model_dir, model_id+'.pth')
        cfg.MODEL.DEVICE = self.device
        cfg.INPUT.MAX_SIZE_TEST = 640
        cfg.INPUT.MIN_SIZE_TEST = 640
        cfg.MODEL.DEPLOY_MODE = True
        cfg.freeze()
        return cfg

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters())

    def _load_model(self, model_id):
        if not self._models.has(model_id):
            cfg = get_cfg()
            add_omdet_v2_turbo_config(cfg)
            cfg.merge_from_file(os.path.join('configs', model_id+'.yaml'))
            cfg = self._init_cfg(cfg, model_id)
            model = Trainer.build_model(cfg)
            self.logger.info("Model:\n{}".format(model))
            DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
            print("Loading a OmDet model {}".format(cfg.MODEL.WEIGHTS))
            model.eval()
            model.to(cfg.MODEL.DEVICE)
            print("Total parameters: {}".format(self.count_parameters(model)))
            self._models.put(model_id, (model, cfg))

        return self._models.get(model_id)

    def inf_predict(self, model_id,
                    data: List,
                    task: Union[str, List],
                    labels: List[str],
                    src_type: str = 'local',
                    conf_threshold: float = 0.5,
                    nms_threshold: float = 0.5
                    ):

        if len(task) == 0:
            raise Exception("Task cannot be empty.")

        model, cfg = self._load_model(model_id)

        resp = []
        flat_labels = labels

        with torch.no_grad():
            for batch in chunks(data, self.batch_size):
                batch_image = self._load_data(src_type, cfg, batch)
                for img in batch_image:
                    img['label_set'] = labels
                    img['tasks'] = task

                batch_y = model(batch_image, score_thresh=conf_threshold, nms_thresh=nms_threshold)

                for z in batch_y:
                    temp = []
                    instances = z['instances'].to('cpu')
                    instances = instances[instances.scores > conf_threshold]

                    for idx, pred in enumerate(zip(instances.pred_boxes, instances.scores, instances.pred_classes)):
                        (x, y, xx, yy), conf, cls = pred
                        conf = float(conf)
                        cls = flat_labels[int(cls)]

                        temp.append({'xmin': int(x),
                                     'ymin': int(y),
                                     'xmax': int(xx),
                                     'ymax': int(yy),
                                     'conf': conf,
                                     'label': cls})
                    resp.append(temp)

        return resp

    def export_onnx(self, model_id, img_tensor, label_feats, task_feats, task_mask, onnx_model_path):

        model, _ = self._load_model(model_id)
        model.to("cpu")
        model.eval()
        inputs = (img_tensor, label_feats, task_feats, task_mask)

        print("start cvt onnx...")
        torch.onnx.export(model,  # model being run
                          inputs,  # model input (or a tuple for multiple inputs)
                          onnx_model_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=17,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['img_tensor', "label_feats", "task_feats", "task_feats"],
                          )