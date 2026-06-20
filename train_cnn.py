"""
Train the EyeStateModel CNN on a labeled eye-image dataset.

Dataset folder structure expected:
    data/
        train/
            open/     <- images of open eyes
            closed/   <- images of closed eyes
        val/
            open/
            closed/

Usage:
    python train_cnn.py --data data/ --epochs 20 --batch 32

After training, a checkpoint is saved to eye_model.pth.
Use cnn_detector.py for real-time inference with the trained model.
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import EyeStateModel

IMG_SIZE = 64


def get_transforms(augment=True):
    base = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if augment:
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomRotation(10),
        ]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)


def train(data_dir, epochs, batch_size, lr, device, save_path):
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), get_transforms(True))
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   get_transforms(False))

    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = EyeStateModel(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

        train_acc  = correct / total
        train_loss = total_loss / total

        # --- val ---
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_dl, desc=f"Epoch {epoch}/{epochs} [val]", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                vc += (out.argmax(1) == labels).sum().item()
                vt += imgs.size(0)

        val_acc = vc / vt
        scheduler.step()

        print(f"Epoch {epoch:3d} | loss {train_loss:.4f} | train {train_acc:.3f} | val {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model": model.state_dict(), "classes": train_ds.classes, "threshold": 0.5},
                save_path,
            )
            print(f"  -> Saved best model (val acc: {val_acc:.3f})")

    print(f"\nDone. Best val acc: {best_val_acc:.3f}  Saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/",       help="Root data dir with train/ and val/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch",  type=int, default=32)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--save",   default="eye_model.pth")
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    train(args.data, args.epochs, args.batch, args.lr, device, args.save)
