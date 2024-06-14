# Grounding DINO 转TensorRT
鉴于不少同学提问想知道我们Paper提到的Grounding DINO的TRT是如何转换，所以在这里简单介绍一下我们之前的转换思路。此外，我们在整理TRT转换时也发现之前的Grounding-DINO-T转换得有点小问题，实际正确转换之后的FP16速度应该为～27FPS。

## pytorch模型 转换成 onnx模型
  原始的Grounding DINO代码稍作修改就能转换成onnx模型， 但是转换成onnx模型后再转换成TensorRT模型时，会有各式各样的花式报错。为了避免onnx 转TensorRT时的报错，必须在转onnx模型时做一些额外的改动。
  
- 注释掉backbone中使用checkpoint的语句
- 将代码中的 NestedTensor 进行改写，不要使用NestedTensor数据结构。NestedTensor主要集中在视觉部分。直接使用Tensor即可
- 将backbone.py 中的Joiner类改写成下面示例。改写后的类要继承nn.Module, 而不是nn.Sequential类。这可能是避免onnx转TensorRT模型出现问题的关键。build_backbone函数里面的部分内容可以移动到改写后的Joint类中
- 将tokenizer 当成数据预处理放在模型的外面，输出直接作为forward函数的输入传入模型
- nested_tensor_from_tensor_list 函数中针对转onnx做的特殊处理需要保留
- 其他一些因为上述改动导致的必要改动

```python
class Joiner(nn.Module):
    def __init__(self):
        self.backbone = xxxx
        self.position_embedding = xxx
    
    def forward(self):
        pass

```


## onnx模型转TensorRT模型
  按照上述建议转出的onnx模型可以流畅的转成TensorRT模型
  
- 建议使用最新版本TensorRT, 真的很快
- 固定输入维度，会有一定的优势。Omdet中关于Grounding DINO 的速度测试都是基于固定的输入维度
- F32 几乎无损， 转换FP16的时候精度损失较大，需要对一些损失较大的层进行额外的处理。Omdet中关于Grounding DINO 的速度测试都是基于FP16模型。FP32 比 FP16 慢 25~30%左右
