import argparse
import random

import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import build_model

from torch.ao.quantization import QConfig
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)


CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_qat_qconfig_int8_int8():
    activation_fake_quant = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False
    )

    weight_fake_quant = FakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        reduce_range=False
    )

    return QConfig(
        activation=activation_fake_quant,
        weight=weight_fake_quant
    )

def get_device():
    # PyTorch eager mode INT8 quantized inference 通常在 CPU 上运行
    return torch.device("cpu")


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])


class QuantRepOptVGGLike(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = build_model(num_classes=num_classes, width_mult=width_mult)
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def fuse_model(model):
    for stage_name in ["stage1", "stage2", "stage3", "stage4"]:
        stage = getattr(model.model, stage_name)
        for m in stage:
            if hasattr(m, "conv") and hasattr(m, "bn") and hasattr(m, "relu"):
                torch.ao.quantization.fuse_modules(
                    m,
                    [["conv", "bn", "relu"]],
                    inplace=True
                )


def build_int8_model(width_mult=1.0):
    model = QuantRepOptVGGLike(num_classes=10, width_mult=width_mult)

    # 1) fuse 前必须 eval
    model.eval()
    model.cpu()
    fuse_model(model)

    # 2) prepare_qat 前必须 train
    model.train()
    # model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
    model.qconfig = get_qat_qconfig_int8_int8()
    torch.ao.quantization.prepare_qat(model, inplace=True)

    # 3) convert 前必须 eval
    model.eval()
    quantized_model = torch.ao.quantization.convert(model, inplace=False)
    return quantized_model


def load_int8_model(weight_path, width_mult=1.0):
    device = get_device()
    model = build_int8_model(width_mult=width_mult)

    ckpt = torch.load(weight_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_one_from_testset(model, data_root, index=None, show=True):
    transform = get_test_transform()

    raw_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=None
    )
    tensor_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )

    if index is None:
        index = random.randint(0, len(raw_dataset) - 1)

    raw_img, true_label = raw_dataset[index]
    input_tensor, _ = tensor_dataset[index]

    input_tensor = input_tensor.unsqueeze(0).to(torch.device("cpu"))

    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_idx].item()

    print(f"Index      : {index}")
    print(f"True label : {CLASSES[true_label]}")
    print(f"Pred label : {CLASSES[pred_idx]}")
    print(f"Confidence : {confidence * 100:.2f}%")

    if show:
        plt.figure(figsize=(4, 4))
        plt.imshow(raw_img)
        plt.axis("off")
        plt.title(
            f"True: {CLASSES[true_label]}\nPred: {CLASSES[pred_idx]} ({confidence * 100:.2f}%)"
        )
        plt.tight_layout()
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Predict one CIFAR-10 test image with INT8 quantized model")
    parser.add_argument(
        "--weight_path",
        type=str,
        required=True,
        help="Path to qat_int8_quantized.pth"
    )
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--index", type=int, default=None, help="Sample index in test set")
    parser.add_argument("--width_mult", type=float, default=1.0)
    parser.add_argument("--no_show", action="store_true", help="Do not show image")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = load_int8_model(args.weight_path, width_mult=args.width_mult)
    predict_one_from_testset(
        model=model,
        data_root=args.data_root,
        index=args.index,
        show=not args.no_show
    )