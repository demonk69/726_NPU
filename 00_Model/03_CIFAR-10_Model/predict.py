# predict.py
import argparse
import random

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import build_model


CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])


def load_model(weight_path, device):
    model = build_model(num_classes=10, pretrained=False, freeze_backbone=False)
    ckpt = torch.load(weight_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_one_from_testset(model, data_root, img_size, index=None, show=True):
    transform = get_test_transform(img_size=img_size)
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

    device = next(model.parameters()).device
    input_tensor = input_tensor.unsqueeze(0).to(device)

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
            f"True: {CLASSES[true_label]}\nPred: {CLASSES[pred_idx]} ({confidence*100:.2f}%)"
        )
        plt.tight_layout()
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Predict one sample from CIFAR-10 test set")
    parser.add_argument("--weight_path", type=str, required=True, help="Path to best.pth")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--index", type=int, default=None, help="Sample index in test set")
    parser.add_argument("--no_show", action="store_true", help="Do not show image")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    model = load_model(args.weight_path, device)
    predict_one_from_testset(
        model=model,
        data_root=args.data_root,
        img_size=args.img_size,
        index=args.index,
        show=not args.no_show
    )