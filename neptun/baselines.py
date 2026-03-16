"""Baseline CNN models and training utilities for CIFAR-10 comparison."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, resnet18


class _YOLOv8Wrapper(nn.Module):
    """Wraps an ultralytics ClassificationModel so forward always returns a tensor.

    In eval mode the underlying model returns (logits, softmax); this wrapper
    extracts only logits so it plugs into standard cross-entropy pipelines.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        out = self.inner(x)
        if isinstance(out, tuple):
            return out[0]
        return out


def _build_yolov8s_cls(num_classes: int) -> nn.Module:
    """Build a YOLOv8s classification model adapted for *num_classes*."""
    from ultralytics import YOLO
    yolo = YOLO('yolov8s-cls.yaml')
    model = yolo.model
    # Replace the final linear layer for the target number of classes
    head = model.model[-1]
    head.linear = nn.Linear(head.linear.in_features, num_classes)
    return _YOLOv8Wrapper(model)


def build_baseline(model_name: str, num_classes: int) -> nn.Module:
    name = model_name.lower()
    if name == 'resnet18':
        return resnet18(num_classes=num_classes)
    if name == 'efficientnet_b0':
        return efficientnet_b0(num_classes=num_classes)
    if name in ('yolov8s', 'yolov8s-cls', 'yolo_small'):
        return _build_yolov8s_cls(num_classes)
    raise ValueError(f"Unsupported baseline model: {model_name}")


def _update_confusion(confusion, pred, target, num_classes):
    idx = (target * num_classes + pred).to(torch.int64)
    bins = torch.bincount(idx, minlength=num_classes * num_classes)
    confusion += bins.view(num_classes, num_classes).cpu()


def _macro_precision_f1_from_confusion(confusion):
    confusion = confusion.to(torch.float32)
    tp = torch.diag(confusion)
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    eps = 1e-12

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    return precision.mean().item(), f1.mean().item()


def train_baseline(model, train_loader, val_loader, epochs, lr, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        val_loss, val_acc = validate_baseline(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

    return history


def validate_baseline(model, loader, device='cpu'):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss += criterion(logits, target).item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return loss / len(loader), correct / total


def evaluate_baseline(model, test_loader, device='cpu'):
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    loss = 0.0
    correct = 0
    total = 0
    confusion = None

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss += criterion(logits, target).item()

            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if confusion is None:
                num_classes = logits.shape[1]
                confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
            _update_confusion(confusion, pred, target, logits.shape[1])

    precision_macro, f1_macro = _macro_precision_f1_from_confusion(confusion)

    return {
        'accuracy': correct / total,
        'loss': loss / len(test_loader),
        'correct': correct,
        'total': total,
        'precision_macro': precision_macro,
        'f1_macro': f1_macro,
    }
