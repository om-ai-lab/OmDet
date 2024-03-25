import torch
from PIL import Image
import requests
import io
import base64
from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy
import numpy as np


def get_output_shape(oldh: int, oldw: int, short_edge_length: int, max_size: int):
    """
    Compute the output size given input size and target short edge length.
    """
    h, w = oldh, oldw
    size = short_edge_length * 1.0
    scale = size / min(h, w)
    if h < w:
        newh, neww = size, scale * w
    else:
        newh, neww = scale * h, size
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


class BaseEngine(object):
    def _load_data(self, src_type, cfg, data, return_transform=False):
        if src_type == 'local':
            image_data = [Image.open(x) for x in data]

        elif src_type == 'url':
            image_data = []
            for x in data:
                temp = Image.open(io.BytesIO(requests.get(x).content))
                image_data.append(temp)

        else:
            raise Exception("Unknown mode {}.".format(src_type))

        input_data = []
        transforms = []
        for x in image_data:
            width, height = x.size
            pil_image = x.resize((cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST), Image.BILINEAR)
            image = convert_PIL_to_numpy(pil_image, cfg.INPUT.FORMAT)

            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            input_data.append({"image": image, "height": height, "width": width})

        if return_transform:
            return input_data, transforms
        else:
            return input_data