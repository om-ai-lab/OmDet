# Grounding DINO to TensorRT Conversion

Given that many people are interested about how to convert Grounding DINO mentioned in our paper to TensorRT, here is a brief introduction to our previous conversion approach. Additionally, while organizing the TRT conversion, we discovered a minor issue with the previous Grounding-DINO-T conversion. The correct FP16 speed after proper conversion should be approximately 27 FPS.

## Converting PyTorch Model to ONNX Model
The original Grounding DINO code requires slight modifications to be converted to an ONNX model. However, when converting the ONNX model to a TensorRT model, various errors may occur. To avoid errors during ONNX to TensorRT conversion, some additional changes must be made when converting to the ONNX model.

- Comment out the statements using checkpoints in the backbone.
- Rewrite the NestedTensor in the code; avoid using the NestedTensor data structure. NestedTensor is mainly concentrated in the visual part. Use Tensor directly instead.
- Rewrite the Joiner class in `backbone.py` as shown in the example below. The rewritten class should inherit from `nn.Module` instead of `nn.Sequential`. This might be the key to avoiding issues when converting the ONNX model to a TensorRT model. Some content in the `build_backbone` function can be moved to the rewritten Joiner class.
- Treat the tokenizer as data preprocessing and place it outside the model; the output should be directly passed as input to the model's forward function.
- The special handling in the `nested_tensor_from_tensor_list` function for ONNX conversion needs to be retained.
- Make other necessary changes due to the above modifications.

```python
class Joiner(nn.Module):
    def __init__(self):
        self.backbone = xxxx
        self.position_embedding = xxx
    
    def forward(self):
        pass
```

## Converting ONNX Model to TensorRT Model
The ONNX model converted according to the above suggestions can be smoothly converted to a TensorRT model.

- It is recommended to use the latest version of TensorRT; it is indeed very fast.
- Fixing the input dimensions can provide certain advantages. The speed tests for Grounding DINO in Omdet are based on fixed input dimensions.
- F32 is almost lossless. When converting to FP16, there is a significant loss of precision, and some layers with substantial losses need extra handling. The speed tests for Grounding DINO in Omdet are based on FP16 models. FP32 is about 25-30% slower than FP16.
