import logging
from collections import Counter

import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, instantiate
from detectron2.data import  build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.utils.analysis import FlopCountAnalysis
from fvcore.nn import flop_count_table

__all__=["do_flop"]

logger = logging.getLogger("detectron2")

def do_flop(cfg):
    if isinstance(cfg, CfgNode):
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TRAIN[0])
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        data_loader = instantiate(cfg.dataloader.test)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()

    counts = Counter()
    total_flops = []
    for idx, data in zip(range(10), data_loader):  # noqa
        flops = FlopCountAnalysis(model, data)
        if idx > 0:
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        counts += flops.by_operator()
        total_flops.append(flops.total())

    logger.info("Flops table computed from only one input sample:\n" + flop_count_table(flops))
    logger.info(
        "Average GFlops for each type of operators:\n"
        + str([(k, v / (idx + 1) / 1e9) for k, v in counts.items()])
    )
    logger.info(
        "Total GFlops: {:.1f}±{:.1f}".format(np.mean(total_flops) / 1e9, np.std(total_flops) / 1e9)
    )
