import io
import base64
import re
from PIL import ImageDraw, Image
import lmdb
from detectron2.data import transforms as T
import logging
from tqdm import tqdm
import os
from detectron2.data import detection_utils as utils
import pickle
import numpy as np
from detectron2.config import CfgNode
from typing import Generator, Sequence
from joblib import Parallel, delayed
import torch
import random

def make_continuous_categories(cats, verbose=True):
    # return a continuous categord_id from 1 to num_classes
    diff_cnt = 0
    for c_id, c in enumerate(cats):
        if c['id'] != c_id+1:
            diff_cnt += 1
        c['id'] = c_id + 1

    if verbose:
        print("Changed {} category_id among {} cats".format(diff_cnt, len(cats)))

    return cats

def is_overlap(a, b):
    if b[1] - b[0] == 0 or a[1] - a[0] == 0:
        return False

    return a[0] <= b[0] < a[1] or b[0] <= a[0] < b[1]


def get_span_embedding(model, tokenizer, sent, spans, layers, device):
    assert len(sent) == len(spans)
    encoded = tokenizer.batch_encode_plus(sent, return_tensors="pt", padding=True)
    encoded = encoded.to(device)
    # token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

    # Only select the tokens that constitute the requested word
    results = []
    for b_id, b_span in enumerate(spans):
        offsets = encoded.encodings[b_id].offsets
        feats = []
        valid_offsets = []
        for t_id, t_span in enumerate(offsets):
            valid = False
            for s in b_span:
                if is_overlap(t_span, s):
                    valid = True
                    break
            if valid:
                feats.append(output[b_id, t_id].view(1, -1))
                valid_offsets.append(t_span)

        if len(feats) == 0:
            raise Exception(f"Sentence '{sent[b_id]}' ({len(sent[b_id])}) cannot find valid span for {b_span}.")

        res = torch.mean(torch.stack(feats, dim=0), dim=0).cpu().tolist()
        results.append(res[0])
    return results


def get_txt_embedding(model, sent):
    txt_embedding = model._text_encode(sent)
    return txt_embedding


def clean_t(x, max_len, rm_sym=True, must_idx=None, return_offset=False):
    """
    rm_sym: remove symbol _
    """
    s_id = 0
    x = x.lower()
    if rm_sym:
        x = x.replace('_', ' ').replace('-', ' ')
        x = ' '.join(x.split())  # remove duplicate space

    if must_idx is not None:
        min_id, max_id = must_idx
        if max_id >= max_len:
            s_id = max(0, min(min_id, int(max_id - (max_len / 2))))
            e_id = min(len(x), int(max_id + (max_len / 2)))
            # print(f"Special cut ({must_idx}): from {s_id} to {e_id} for sent of len {len(x)}")
            x = x[s_id:e_id]
    else:
        x = x[0:max_len]
    if return_offset:
        return x, s_id
    else:
        return x

def sample_true(prob):
    if prob <= 0:
        return False
    generated_neg_prob = random.random()
    valid = generated_neg_prob < prob
    return valid

def rm_duplicates(input_list, keep_order=False):
    if not keep_order:
        return list(set(input_list))

    # Create an empty set to store the items that have been seen
    seen = set()

    # Create an empty list to store the result
    result = []

    # Iterate over the input list
    for item in input_list:
        # If the item is not already in the seen set, add it to the result list
        if item not in seen:
            result.append(item)

        # Add the item to the seen set
        seen.add(item)

    # Return the result list
    return result


def chunks(l: Sequence, n: int = 5) -> Generator[Sequence, None, None]:
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def encode_dump_text(model, feat_path, text_vocab, batch_size):
    text_keys = []
    for block in tqdm(chunks(text_vocab, n=batch_size)):
        block_feats = []
        block_keys = []
        for batch in chunks(block, n=500):
            batch_fs = get_txt_embedding(model, batch)
            batch_keys = batch
            block_feats.extend(batch_fs)
            block_keys.extend(batch_keys)

        text_keys.extend(block_keys)
        write_lmdb_from_id_data_pairs(
            id_data_pairs=[(key, embed) for key, embed in zip(block_keys, block_feats)],
            lmdb_save_dir=feat_path
        )
    return text_keys


def cropbox(xmin, ymin, xmax, ymax, img_size, ratio=1.5, make_square=False):
    if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
        raise Exception
    w, h = img_size
    if xmin > w or ymin > h or xmax > w or ymax > h:
        raise Exception

    xc = xmin + (xmax - xmin) / 2
    yc = ymin + (ymax - ymin) / 2
    w = xmax - xmin
    h = ymax - ymin
    nw = w * ratio
    nh = h * ratio

    if make_square:
        if nw > nh:
            nh = nw
        else:
            nw = nh

    nxmin = max(xc - (nw / 2), 0)
    nymin = max(yc - (nh / 2), 0)

    nxmax = min(xc + (nw / 2), img_size[0])
    nymax = min(yc + (nh / 2), img_size[1])

    return nxmin, nymin, nxmax, nymax


def image_to_base64(img):
    output_buffer = io.BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str


def base64_to_image(base64_str):
    return Image.open(io.BytesIO(base64.b64decode(base64_str)))


def draw_bounding_box_on_image(image, xmin, ymin, xmax, ymax,
                               color='red',
                               text='',
                               thickness=4):
    draw = ImageDraw.Draw(image)
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=thickness)
    draw.text((xmin, ymin), text)
    return image


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    # tfm_gens.append(T.Resize(min_size))
    if is_train:
        logger = logging.getLogger(__name__)
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


def jp(a, b):
    return os.path.join(a, b)


def check_img(i, img_root):
    # i['file_name'] = i['file_name'].split('/')[-1]
    try:
        iimage = utils.read_image(jp(img_root, i["file_name"]), format='RGB')
        utils.check_image_size(i, iimage)

    except Exception as e:
        print("BAD D2 IMG", i)
        if 'image_id' in i:
            return i['image_id']
        else:
            return i['id']

    return None


def fix_img_size(i, img_root):
    try:
        if not "file_name" in i:
            i["file_name"] = i["coco_url"].split("/")[-1]
        img = Image.open(jp(img_root, i['file_name']))
        w, h = img.size
        if i['width'] != w or i['height'] != h:
            print("Found image {} with wrong size.\n".format(i['id']))
            i['width'] = w
            i['height'] = h

        return i
    except Exception as e:
        print("BAD IMG", i, e)
        return None


def fix_data(img_root, data):
    if type(data) is dict:
        num_imgs = len(data['images'])
        data['images'] = Parallel(n_jobs=15, backend='threading')(
            delayed(fix_img_size)(i, img_root) for i in tqdm(data['images']))
        data['images'] = [i for i in data['images'] if i is not None]
        print("First stage image fixing go from {} to {}".format(num_imgs, len(data['images'])))

        bad_ids = Parallel(n_jobs=15, backend='threading')(delayed(check_img)(i, img_root) for i in tqdm(data['images']))
        bad_ids = [x for x in set(bad_ids) if x is not None]
        print("Found {} bad images with D2 checking".format(len(bad_ids)))
        data['images'] = [d for d in data['images'] if d['id'] not in bad_ids]
        print("Images go from {} to {}".format(num_imgs, len(data['images'])))

        prev_anno_size = len(data['annotations'])
        valid_imgs = {i['id'] for i in data['images']}
        data['annotations'] = [d for d in data['annotations'] if d['image_id'] in valid_imgs]
        print("Anno go from {} to {} after fixing.".format(prev_anno_size, len(data['annotations'])))
    else:
        num_imgs = len(data)
        data = Parallel(n_jobs=15, backend='threading')(delayed(fix_img_size)(i, img_root) for i in tqdm(data))
        data = [i for i in data if i is not None]
        print("First stage image fixing go from {} to {}".format(num_imgs, len(data)))

        bad_ids = Parallel(n_jobs=15, backend='threading')(delayed(check_img)(i, img_root) for i in tqdm(data))
        bad_ids = [x for x in set(bad_ids) if x is not None]
        print("Found {} bad images with D2 checking".format(len(bad_ids)))
        data = [d for d in data if d['id'] not in bad_ids]
        print("Images go from {} to {}".format(num_imgs, len(data)))
    return data


def convert_cfg_to_dict(cfg_node, key_list):
    if not isinstance(cfg_node, CfgNode):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_cfg_to_dict(v, key_list + [k])
        return cfg_dict


def flatten_json(json_file):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '.')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '.')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(json_file)
    return out


def convert_to_float(value):
    if isinstance(value, float):
        return value
    try:  # try pytorch
        return value.item()
    except:
        try:  # try numpy
            print(value.dtype)
            return np.asscalar(value)
        except:
            raise ValueError('do not know how to convert this number {} to float'.format(value))


def remove_punctuation(text: str) -> str:
    punct = ['|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^',
             '\'', '\"', '’', '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
             ]
    for p in punct:
        text = text.replace(p, '')
    return text.strip()