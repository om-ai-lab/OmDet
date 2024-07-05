import os

from omdet.inference.det_engine import DetEngine
from omdet.utils.plots import Annotator
from PIL import Image
import numpy as np

if __name__ == "__main__":
    engine = DetEngine(batch_size=1, device='cuda')
    img_paths = ['./sample_data/000000574769.jpg']       # path of images
    labels = ["person", "cat", "orange"]          # labels to be predicted
    prompt = 'Detect {}.'.format(','.join(labels))        # prompt of detection task, use "Detect {}." as default

    res = engine.inf_predict('OmDet-Turbo_tiny_SWIN_T',    # prefix name of the pretrained checkpoints
                           task=prompt,
                           data=img_paths,
                           labels=labels,
                           src_type='local',                     # type of the image_paths, "local"/"url"
                           conf_threshold=0.30,
                           nms_threshold=0.5
                          )
    print(res)

    out_folder = './outputs'
    for idx, img_path in enumerate(img_paths):
        im = Image.open(img_path)
        a = Annotator(np.ascontiguousarray(im), font_size=12, line_width=1, pil=True, font='sample_data/simsun.ttc')
        for R in res[idx]:
            a.box_label([R['xmin'], R['ymin'], R['xmax'], R['ymax']],
                        label=f"{R['label']} {str(int(R['conf'] * 100))}%",
                        color='red')

        if not os.path.exists(out_folder):
            os.mkdir(out_folder)

        image = a.result()
        img = Image.fromarray(image)
        img.save('outputs/'+img_path.split('/')[-1])