from omdet.inference.det_engine import DetEngine
import torch

if __name__ == "__main__":

    model_dir = "./resources"
    img_tensor =  torch.rand(1, 3, 640, 640) #
    label_feats = torch.rand(80, 1, 512) # 80 is cls num, 512 is clip dim
    task_feats = torch.rand(77, 1, 512)  # 77 is task dim
    task_mask = torch.rand(1, 77)

    engine = DetEngine(model_dir=model_dir, batch_size=1, device='cpu')
    onnx_model_path = "./omdet.onnx"
    engine.export_onnx('OmDet-Turbo_tiny_SWIN_T', img_tensor, label_feats, task_feats, task_mask, onnx_model_path)

