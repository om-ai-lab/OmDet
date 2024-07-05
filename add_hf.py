from omdet.inference.det_engine import DetEngine
from omdet.omdet_v2_turbo.detector import OmDetV2Turbo


if __name__ == "__main__":
    engine = DetEngine(batch_size=1, device='cuda')
    img_paths = ['./sample_data/000000574769.jpg']       # path of images
    labels = ["person", "cat", "orange"]          # labels to be predicted
    prompt = 'Detect {}.'.format(','.join(labels))        # prompt of detection task, use "Detect {}." as default

    model_id = 'OmDet-Turbo_tiny_SWIN_T'
    model, cfg = engine._load_model(model_id)

    # push to hub
    model.push_to_hub("nielsr/omde-v2-turbo-tiny-swin-tiny")

    # reload
    model = OmDetV2Turbo.from_pretrained("nielsr/omde-v2-turbo-tiny-swin-tiny")