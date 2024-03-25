import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
import platform
import math

def is_writeable(dir, test=False):
    # Return True if directory has write permissions, test opening a file with write permissions if test=True
    if test:  # method 1
        file = Path(dir) / 'tmp.txt'
        try:
            with open(file, 'w'):  # open file with write permissions
                pass
            file.unlink()  # remove file
            return True
        except IOError:
            return False
    else:  # method 2
        return os.access(dir, os.R_OK)  # possible issues on Windows

def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
    # Return path of user configuration directory. Prefer environment variable if exists. Make dir if required.
    env = os.getenv(env_var)
    if env:
        path = Path(env)  # use environment variable
    else:
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}  # 3 OS dirs
        path = Path.home() / cfg.get(platform.system(), '')  # OS-specific config dir
        path = (path if is_writeable(path) else Path('/tmp')) / dir  # GCP and AWS lambda fix, only /tmp is writeable
    path.mkdir(exist_ok=True)  # make if required
    return path

# Settings
CONFIG_DIR = user_config_dir()  # Ultralytics settings dir
RANK = int(os.getenv('RANK', -1))
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only

def check_font(font='Arial.ttf', size=10):
    # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
    font = Path(font)
    font = font if font.exists() else (CONFIG_DIR / font.name)
    try:
        return ImageFont.truetype(str(font) if font.exists() else font.name, size)
    except Exception as e:  # download if missing
        url = "https://ultralytics.com/assets/" + font.name
        print(f'Downloading {url} to {font}...')
        torch.hub.download_url_to_file(url, str(font), progress=False)
        return ImageFont.truetype(str(font), size)

def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters?
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)


class Annotator:
    # if RANK in (-1, 0):
    #     check_font()  # download TTF if necessary

    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=True):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.pil = pil
        self.offset = 0
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.im = ImageOps.expand(self.im, border=self.offset, fill=(255, 255, 255))
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_font(font, size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
            self.fh = 5  # font height
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def _offset_box(self, box):
        return (np.array(box)+self.offset).tolist()

    def draw_arrow(self, ptA, ptB, width=1, color=(0, 255, 0)):
        """Draw line from ptA to ptB with arrowhead at ptB"""
        # Get drawing context
        # Draw the line without arrows
        self.draw.line((ptA, ptB), width=width, fill=color)

        # Now work out the arrowhead
        # = it will be a triangle with one vertex at ptB
        # - it will start at 95% of the length of the line
        # - it will extend 8 pixels either side of the line
        x0, y0 = ptA
        x1, y1 = ptB
        # Now we can work out the x,y coordinates of the bottom of the arrowhead triangle
        xb = 0.95 * (x1 - x0) + x0
        yb = 0.95 * (y1 - y0) + y0

        # Work out the other two vertices of the triangle
        # Check if line is vertical
        if x0 == x1:
            vtx0 = (xb - 5, yb)
            vtx1 = (xb + 5, yb)
        # Check if line is horizontal
        elif y0 == y1:
            vtx0 = (xb, yb + 5)
            vtx1 = (xb, yb - 5)
        else:
            alpha = math.atan2(y1 - y0, x1 - x0) - 90 * math.pi / 180
            a = 8 * math.cos(alpha)
            b = 8 * math.sin(alpha)
            vtx0 = (xb + a, yb + b)
            vtx1 = (xb - a, yb - b)

        # draw.point((xb,yb), fill=(255,0,0))    # DEBUG: draw point of base in red - comment out draw.polygon() below if using this line
        # im.save('DEBUG-base.png')              # DEBUG: save

        # Now draw the arrowhead triangle
        self.draw.polygon([vtx0, vtx1, ptB], fill=color)

    def box_label(self, box, label='', sub_label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        box = self._offset_box(box)
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = 2, 2 # text width
                self.draw.rectangle([box[0], box[1] - self.fh, box[0] + w + 1, box[1] + 1], fill=color)
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((box[0], box[1] - h), label+'\n'+sub_label, fill=txt_color, font=self.font)
        else:  # cv2
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, c1, c2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]
                c2 = c1[0] + w, c1[1] - h - 3
                cv2.rectangle(self.im, c1, c2, color, -1, cv2.LINE_AA)  # filled
                ft = cv2.freetype.createFreeType2()
                ft.putText(self.im, label+'\n'+sub_label, (c1[0], c1[1] - 2), 0, self.lw / 3, txt_color, thickness=tf,
                            lineType=cv2.LINE_AA)

    def tuple_label(self, src_box, dest_box, label='', src_color='red', dest_color='blue', txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        src_box = self._offset_box(src_box)
        dest_box = self._offset_box(dest_box)

        if self.pil or not is_ascii(label):
            self.draw.rectangle(src_box, width=self.lw, outline=src_color)  # box
            self.draw.rectangle(dest_box, width=self.lw, outline=dest_color)  # box
            src_c = (int((src_box[2]+src_box[0])/2), int((src_box[3]+src_box[1])/2))
            dest_c = (int((dest_box[2]+dest_box[0])/2), int((dest_box[3]+dest_box[1])/2))
            c_c = [(src_c[0]+dest_c[0])/2, (src_c[1]+dest_c[1])/2]
            # self.draw.line(xy=[src_c, dest_c], fill='green')
            self.draw_arrow(src_c, dest_c, color='green', width=2)

            if label:
                w, h = self.font.getsize(label)  # text width
                self.draw.rectangle([c_c[0], c_c[1] - self.fh, c_c[0] + w + 1, c_c[1] + 1], fill='green')
                self.draw.text((c_c[0], c_c[1] - h), label, fill=txt_color, font=self.font)

        else:  # cv2
           raise Exception("CV2 is not supported yet")

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        # Add text to image (PIL-only)
        w, h = self.font.getsize(text)  # text width, height
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)

















