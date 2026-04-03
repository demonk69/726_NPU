# qat.py
import os
import csv
import json
import math
import copy
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as tq
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import build_model


CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device():
    # QAT训练本身通常在GPU上进行fake quant
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])
    return train_transform, test_transform


def build_dataloaders(data_root="./data", batch_size=128, num_workers=4, val_ratio=0.1):
    train_tf, test_tf = get_transforms()

    full_train_set = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_tf
    )

    full_train_set_for_val = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=test_tf
    )

    test_set = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_tf
    )

    total_len = len(full_train_set)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        range(total_len), [train_len, val_len], generator=generator
    )

    train_indices = train_subset.indices
    val_indices = val_subset.indices

    train_set = torch.utils.data.Subset(full_train_set, train_indices)
    val_set = torch.utils.data.Subset(full_train_set_for_val, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader, test_loader


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return self.should_stop


def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct, total


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Eval"):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)

        running_loss += loss.item() * total
        running_correct += correct
        running_total += total

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return (
        running_loss / running_total,
        100.0 * running_correct / running_total,
        all_preds,
        all_labels
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    pbar = tqdm(loader, desc="QAT Train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        correct, total = accuracy_from_logits(logits, labels)
        running_loss += loss.item() * total
        running_correct += correct
        running_total += total

        pbar.set_postfix({
            "loss": f"{running_loss / running_total:.4f}",
            "acc": f"{100.0 * running_correct / running_total:.2f}%"
        })

    return running_loss / running_total, 100.0 * running_correct / running_total


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
    """
    融合 Conv + BN + ReLU
    注意：融合前模型必须处于 eval() 模式
    """
    for stage_name in ["stage1", "stage2", "stage3", "stage4"]:
        stage = getattr(model.model, stage_name)
        for m in stage:
            # 只对 ConvBNReLU 做融合
            if hasattr(m, "conv") and hasattr(m, "bn") and hasattr(m, "relu"):
                torch.ao.quantization.fuse_modules(
                    m,
                    [["conv", "bn", "relu"]],
                    inplace=True
                )


def prepare_qat_model(fp32_ckpt_path, device, width_mult=1.0):
    qat_model = QuantRepOptVGGLike(num_classes=10, width_mult=width_mult)

    ckpt = torch.load(fp32_ckpt_path, map_location="cpu")
    fp32_state_dict = ckpt["model_state_dict"]
    qat_model.model.load_state_dict(fp32_state_dict)

    # 1) 先 eval，给 fuse 用
    qat_model.eval()
    qat_model.cpu()
    fuse_model(qat_model)

    # 2) 再切回 train，给 prepare_qat 用
    qat_model.train()

    # 3) 设置 qconfig 并 prepare_qat
    qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
    torch.ao.quantization.prepare_qat(qat_model, inplace=True)

    # 4) 最后再搬到训练设备
    qat_model.to(device)
    return qat_model


def save_history_plot(history, save_dir):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("QAT Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "qat_loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("QAT Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "qat_acc_curve.png"), dpi=200)
    plt.close()


def save_history_csv(history, save_path):
    keys = list(history.keys())
    rows = zip(*[history[k] for k in keys])
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)


def main(args):
    seed_everything(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio
    )

    model = prepare_qat_model(
        fp32_ckpt_path=args.fp32_ckpt,
        device=device,
        width_mult=args.width_mult
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_val_acc = 0.0
    early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    best_model_state = None

    for epoch in range(1, args.epochs + 1):
        print(f"\nQAT Epoch [{epoch}/{args.epochs}]")

        # 通常QAT后期冻结BN统计更稳一点
        if epoch == args.freeze_bn_epoch:
            print("Freezing observers and batch norm stats...")
            model.apply(tq.disable_observer)
            model.apply(tq.freeze_bn_stats)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, desc="QAT Val")
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_model_state,
                    "best_val_acc": best_val_acc,
                    "history": history,
                    "args": vars(args)
                },
                save_dir / "qat_best_fakequant.pth"
            )
            print(f"Best QAT fake-quant model saved. Best Val Acc = {best_val_acc:.2f}%")

        if early_stopper.step(val_acc):
            print(f"Early stopping triggered. patience={args.patience}")
            break

    print("\nQAT training finished.")
    print(f"Best Val Acc: {best_val_acc:.2f}%")

    # 载入最佳QAT参数
    model.load_state_dict(best_model_state)
    model.to("cpu")
    model.eval()

    # 转成真正量化模型
    quantized_model = tq.convert(model, inplace=False)

    # 保存量化模型
    torch.save(
        {
            "model_state_dict": quantized_model.state_dict(),
            "best_val_acc": best_val_acc,
            "args": vars(args)
        },
        save_dir / "qat_int8_quantized.pth"
    )

    # 量化模型测试通常在CPU上
    test_loss, test_acc, _, _ = evaluate(quantized_model, test_loader, criterion, torch.device("cpu"), desc="INT8 Test")
    print(f"\nINT8 Quantized Test Loss: {test_loss:.4f}")
    print(f"INT8 Quantized Test Acc : {test_acc:.2f}%")

    result_json = {
        "best_qat_val_acc": best_val_acc,
        "int8_test_loss": test_loss,
        "int8_test_acc": test_acc
    }
    with open(save_dir / "qat_results.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    save_history_plot(history, str(save_dir))
    save_history_csv(history, str(save_dir / "qat_history.csv"))

    print(f"\nAll QAT outputs saved to: {save_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="QAT for INT8-friendly RepOpt-VGG-style CNN")
    parser.add_argument("--fp32_ckpt", type=str, required=True, help="Path to FP32 best.pth")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./runs/cifar10_repopt_vgglike_qat")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--freeze_bn_epoch", type=int, default=20)
    parser.add_argument("--width_mult", type=float, default=1.0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)