# model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    构建适配 CIFAR-10 的 ResNet18
    - 支持 ImageNet 预训练
    - 将最后全连接层改为 10 类
    - 可选冻结主干网络
    """
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        model = resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    return model


if __name__ == "__main__":
    net = build_model()
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print("Output shape:", y.shape)  # [2, 10]