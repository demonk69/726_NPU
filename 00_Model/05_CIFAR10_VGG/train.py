# train.py
import os
import csv
import json
import math
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    def __init__(self, patience=20, min_delta=0.0):
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


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_per_class_accuracy(cm):
    result = {}
    for i in range(cm.shape[0]):
        total = cm[i].sum()
        acc = 0.0 if total == 0 else 100.0 * cm[i, i] / total
        result[CLASSES[i]] = acc
    return result


def save_confusion_matrix_figure(cm, save_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45)
    plt.yticks(tick_marks, CLASSES)

    threshold = cm.max() / 2 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_history_plot(history, save_dir):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "acc_curve.png"), dpi=200)
    plt.close()


def save_history_csv(history, save_path):
    keys = list(history.keys())
    rows = zip(*[history[k] for k in keys])
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)


def save_checkpoint(state, save_path):
    torch.save(state, save_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_acc = ckpt.get("best_val_acc", 0.0)
    history = ckpt.get("history", None)
    return start_epoch, best_val_acc, history


@torch.no_grad()
def visualize_predictions(model, loader, device, save_path, num_images=16):
    model.eval()

    images_all = []
    labels_all = []
    preds_all = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        images_all.append(images.cpu())
        labels_all.append(labels.cpu())
        preds_all.append(preds.cpu())

        if sum(x.size(0) for x in images_all) >= num_images:
            break

    images = torch.cat(images_all, dim=0)[:num_images]
    labels = torch.cat(labels_all, dim=0)[:num_images]
    preds = torch.cat(preds_all, dim=0)[:num_images]

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    rows = int(math.sqrt(num_images))
    cols = math.ceil(num_images / rows)

    plt.figure(figsize=(3 * cols, 3 * rows))
    for i in range(num_images):
        img = images[i] * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        color = "green" if preds[i] == labels[i] else "red"
        plt.title(f"T:{CLASSES[labels[i]]}\nP:{CLASSES[preds[i]]}", color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


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

    model = build_model(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and args.amp) else None

    start_epoch = 1
    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    latest_ckpt = save_dir / "latest.pth"
    best_ckpt = save_dir / "best.pth"

    if args.resume and latest_ckpt.exists():
        print(f"Resuming from: {latest_ckpt}")
        start_epoch, best_val_acc, loaded_history = load_checkpoint(
            checkpoint_path=str(latest_ckpt),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device
        )
        if loaded_history is not None:
            history = loaded_history

    early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, desc="Val")
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

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val_acc": best_val_acc,
            "history": history,
            "args": vars(args)
        }
        save_checkpoint(checkpoint_state, latest_ckpt)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_state["best_val_acc"] = best_val_acc
            save_checkpoint(checkpoint_state, best_ckpt)
            print(f"Best model saved. Best Val Acc = {best_val_acc:.2f}%")

        if early_stopper.step(val_acc):
            print(f"Early stopping triggered. patience={args.patience}")
            break

    print("\nTraining finished.")
    print(f"Best Val Acc: {best_val_acc:.2f}%")

    print("\nLoading best model for final test...")
    load_checkpoint(str(best_ckpt), model, device=device)

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, desc="Test"
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc : {test_acc:.2f}%")

    cm = compute_confusion_matrix(test_labels, test_preds, num_classes=10)
    per_class_acc = compute_per_class_accuracy(cm)

    print("\nPer-class Accuracy:")
    for cls_name, acc in per_class_acc.items():
        print(f"{cls_name:>10s}: {acc:.2f}%")

    save_history_plot(history, str(save_dir))
    save_history_csv(history, str(save_dir / "history.csv"))
    save_confusion_matrix_figure(cm, str(save_dir / "confusion_matrix.png"))
    visualize_predictions(
        model=model,
        loader=test_loader,
        device=device,
        save_path=str(save_dir / "sample_predictions.png"),
        num_images=16
    )

    result_json = {
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "per_class_accuracy": per_class_acc
    }

    with open(save_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    print(f"\nAll outputs saved to: {save_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train VGG-Like CNN on CIFAR-10")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./runs/cifar10_vgglike")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)