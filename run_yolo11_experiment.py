"""YOLO11s-cls baseline sweep on CIFAR-10.

Mirrors the YOLOv8s sweep: tests multiple epoch/LR combos.
Results saved to outputs/experiments/yolo_vs_dnbn_20260316/
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neptun.baselines import build_baseline, evaluate_baseline, train_baseline
from neptun.datasets import get_dataset


def main():
    device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    output_dir = Path("outputs/experiments/yolo_vs_dnbn_20260316")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, input_shape, output_dim = get_dataset(
        "cifar10", batch_size=64
    )

    configs = [
        (10, 0.001),
        (20, 0.001),
        (30, 0.001),
        (20, 0.0005),
    ]

    all_rows = []

    for epochs, lr in configs:
        tag = f"yolo11s_e{epochs}_lr{lr}"
        print("\n" + "=" * 72)
        print(f"Training YOLO11s-cls | epochs={epochs} lr={lr}")
        print("=" * 72)

        model = build_baseline("yolo11s", output_dim)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Params: {total_params:,}")

        t0 = time.time()
        history = train_baseline(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)
        elapsed = time.time() - t0

        metrics = evaluate_baseline(model, test_loader, device=device)

        print(f"\n=== {tag} Results ===")
        print(f"Test Accuracy:  {metrics['accuracy']:.6f}")
        print(f"Test F1 macro:  {metrics['f1_macro']:.6f}")
        print(f"Best Val Acc:   {max(history['val_acc']):.6f}")
        print(f"Time:           {elapsed:.1f}s")

        run_dir = output_dir / f"baseline_{tag}"
        run_dir.mkdir(exist_ok=True)

        row = {
            "model": "yolo11s-cls",
            "tag": tag,
            "params": total_params,
            "epochs": epochs,
            "learning_rate": lr,
            "test_accuracy": metrics["accuracy"],
            "test_f1_macro": metrics["f1_macro"],
            "test_precision_macro": metrics["precision_macro"],
            "test_loss": metrics["loss"],
            "best_val_acc": max(history["val_acc"]),
            "duration_sec": elapsed,
        }
        all_rows.append(row)

        csv_path = run_dir / "results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)

        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # Write aggregate CSV
    agg_path = output_dir / "yolo11_baseline_sweep.csv"
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved aggregate: {agg_path}")

    # Summary
    best = max(all_rows, key=lambda r: r["test_accuracy"])
    print(f"\n{'='*72}")
    print(f"Best YOLO11s config: {best['tag']}")
    print(f"  Accuracy: {best['test_accuracy']:.6f}")
    print(f"  F1:       {best['test_f1_macro']:.6f}")
    print(f"  Params:   {best['params']:,}")


if __name__ == "__main__":
    main()
