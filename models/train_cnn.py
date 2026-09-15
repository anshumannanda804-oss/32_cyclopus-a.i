"""
Train the CycloneImageClassifier CNN on synthetic satellite IR images.

Usage:
    1. First run generate_training_data.py to create images.
    2. Then run this script:
       python models/train_cnn.py

Saves:
    models/cyclone_cnn.pth   — PyTorch state dict
    models/cyclone_cnn.pkl   — Pickle wrapper (state dict + class labels)
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Ensure models/ is on path
sys.path.insert(0, os.path.dirname(__file__))
from cnn_model import CycloneImageClassifier, INPUT_SIZE, CLASS_LABELS

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "training_data")
MODEL_DIR = os.path.dirname(__file__)
WEIGHTS_PATH = os.path.join(MODEL_DIR, "cyclone_cnn.pth")
PKL_PATH = os.path.join(MODEL_DIR, "cyclone_cnn.pkl")

EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.8
SEED = 42


def train():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── Data Transforms ───────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # ── Dataset ────────────────────────────────────────────
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    n_total = len(full_dataset)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val = n_total - n_train

    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    # Apply val transforms (no augmentation) to validation set
    val_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Dataset: {n_total} images  |  Train: {n_train}  |  Val: {n_val}")
    print(f"  Classes: {full_dataset.classes}")

    # ── Model ──────────────────────────────────────────────
    classifier = CycloneImageClassifier()
    model = classifier.get_raw_model().to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader), epochs=EPOCHS
    )

    # ── Training Loop ─────────────────────────────────────
    print("\n  " + "─" * 55)
    print(f"  {'Epoch':>5}  {'Train Loss':>11}  {'Train Acc':>10}  {'Val Acc':>8}  {'Time':>6}")
    print("  " + "─" * 55)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total * 100

        # ── Validation ─────────────────────────────────────
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100 if val_total > 0 else 0
        elapsed = time.time() - t0

        print(f"  {epoch:5d}  {train_loss:11.4f}  {train_acc:9.2f}%  {val_acc:7.2f}%  {elapsed:5.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # ── Save Weights ───────────────────────────────────────
    print("\n  " + "─" * 55)
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")

    # Save .pth state dict
    classifier.model = model
    classifier.save_weights(WEIGHTS_PATH)
    print(f"  ✅ Saved PyTorch weights: {WEIGHTS_PATH}")

    # Save .pkl wrapper
    classifier.save_as_pkl(PKL_PATH)
    print(f"  ✅ Saved Pickle model:    {PKL_PATH}")

    return classifier


if __name__ == "__main__":
    print("=" * 60)
    print("Cyclone AI — CNN Training Pipeline")
    print("=" * 60)
    train()
    print("\n  🎉 Training complete!")
