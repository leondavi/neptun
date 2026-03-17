"""YOLO vs DNBN comparison experiment on STL-10.

Fully automated — no user input required.

Steps:
1. Train YOLOv8s-cls baseline at multiple epoch/LR configs
2. Train YOLO11s-cls baseline at the same configs
3. Run DNBN 8-node sweep with decreasing M/C
4. Save all results to outputs/experiments/stl10_yolo_vs_dnbn_<date>/
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
from neptun.config import load_sys_config
from neptun.datasets import get_dataset
from neptun.evaluator import evaluate_system
from neptun.system import DNBNSystem
from neptun.trainer import Trainer


def run_yolo_sweep(model_key, model_label, configs, train_loader, val_loader,
                   test_loader, output_dim, output_dir, device):
    """Train a YOLO model across multiple (epochs, lr) configs."""
    all_rows = []
    for epochs, lr in configs:
        tag = f"{model_key}_e{epochs}_lr{lr}"
        print("\n" + "=" * 72)
        print(f"Training {model_label} | epochs={epochs} lr={lr}")
        print("=" * 72)

        model = build_baseline(model_key, output_dim)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Params: {total_params:,}")

        t0 = time.time()
        history = train_baseline(model, train_loader, val_loader,
                                 epochs=epochs, lr=lr, device=device)
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
            "model": model_label,
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

        with open(run_dir / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)

        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    return all_rows


def run_dnbn_sweep(dnbn_configs, train_loader, val_loader, test_loader,
                   input_shape, output_dim, output_dir, device_str):
    """Train DNBN system across multiple M/C configs."""
    device = torch.device(device_str)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    all_rows = []

    for i, config_path in enumerate(dnbn_configs, 1):
        run_name = Path(config_path).stem
        run_dir = output_dir / run_name
        run_dir.mkdir(exist_ok=True)

        print(f"\n{'='*72}")
        print(f"[{i}/{len(dnbn_configs)}] DNBN {run_name}")
        print(f"{'='*72}")

        with open(os.path.join(repo_root, config_path)) as f:
            sys_cfg = json.load(f)

        for node_cfg in sys_cfg["nodes"].values():
            dnbn_path = node_cfg["config"]
            if not os.path.isabs(dnbn_path):
                dnbn_path = os.path.join(repo_root, dnbn_path)
            with open(dnbn_path) as f:
                node_cfg["params"] = json.load(f)

        sys_cfg["training"]["epochs"] = 10

        system = DNBNSystem(sys_cfg, input_channels=input_shape[0],
                            output_dim=output_dim)
        total_params = sum(p.numel() for p in system.parameters())
        first_node_params = next(iter(sys_cfg["nodes"].values()))["params"]
        mc = first_node_params.get("C", 256)

        print(f"M/C: {mc}, Nodes: 8, Params: {total_params:,}")

        t0 = time.time()
        trainer = Trainer(system, train_loader, val_loader, sys_cfg)
        history = trainer.train(device)
        eval_results, comm_stats = evaluate_system(system, test_loader, device)
        elapsed = time.time() - t0

        ensemble = eval_results["ensemble"]
        print(f"Results: acc={ensemble['accuracy']:.6f} "
              f"f1={ensemble['f1_macro']:.6f} "
              f"params={total_params:,} time={elapsed:.1f}s")

        row = {
            "run_name": run_name,
            "M_C": mc,
            "nodes": 8,
            "total_params": total_params,
            "epochs": 10,
            "test_accuracy": ensemble["accuracy"],
            "test_f1_macro": ensemble["f1_macro"],
            "test_precision_macro": ensemble["precision_macro"],
            "test_loss": ensemble["loss"],
            "best_val_acc": max(history["val_acc"]),
            "duration_sec": elapsed,
        }
        all_rows.append(row)

        with open(run_dir / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)

    return all_rows


def main():
    device = ("mps" if hasattr(torch.backends, "mps")
              and torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path("outputs/experiments/stl10_yolo_vs_dnbn_20260316")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, input_shape, output_dim = get_dataset(
        "stl10", batch_size=64
    )

    # --- Hyperparameter grid (shared by all YOLO baselines) ---
    hp_grid = [
        (10, 0.001),
        (20, 0.001),
        (30, 0.001),
        (20, 0.0005),
    ]

    # ==================== YOLOv8s-cls ====================
    print("\n" + "#" * 72)
    print("# YOLOv8s-cls Baseline Sweep")
    print("#" * 72)
    yolov8_rows = run_yolo_sweep(
        "yolov8s", "yolov8s-cls", hp_grid,
        train_loader, val_loader, test_loader, output_dim, output_dir, device,
    )

    agg = output_dir / "yolov8_baseline_sweep.csv"
    with open(agg, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(yolov8_rows[0].keys()))
        w.writeheader()
        w.writerows(yolov8_rows)

    # ==================== YOLO11s-cls ====================
    print("\n" + "#" * 72)
    print("# YOLO11s-cls Baseline Sweep")
    print("#" * 72)
    yolo11_rows = run_yolo_sweep(
        "yolo11s", "yolo11s-cls", hp_grid,
        train_loader, val_loader, test_loader, output_dim, output_dir, device,
    )

    agg = output_dir / "yolo11_baseline_sweep.csv"
    with open(agg, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(yolo11_rows[0].keys()))
        w.writeheader()
        w.writerows(yolo11_rows)

    # ==================== DNBN 8-node sweep ====================
    print("\n" + "#" * 72)
    print("# DNBN 8-Node Neuron Sweep")
    print("#" * 72)
    dnbn_configs = [
        "configs/sys_dnbn_cifar10_8node_m48c48_tuned.json",
        "configs/sys_dnbn_cifar10_8node_m32c32_tuned.json",
        "configs/sys_dnbn_cifar10_8node_m24c24_tuned.json",
        "configs/sys_dnbn_cifar10_8node_m16c16_tuned.json",
    ]
    dnbn_rows = run_dnbn_sweep(
        dnbn_configs, train_loader, val_loader, test_loader,
        input_shape, output_dim, output_dir, device,
    )

    agg = output_dir / "dnbn_sweep_results.csv"
    with open(agg, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dnbn_rows[0].keys()))
        w.writeheader()
        w.writerows(dnbn_rows)

    # ==================== Final Summary ====================
    print("\n" + "=" * 72)
    print("EXPERIMENT COMPLETE — Summary")
    print("=" * 72)

    best_v8 = max(yolov8_rows, key=lambda r: r["test_accuracy"])
    best_11 = max(yolo11_rows, key=lambda r: r["test_accuracy"])
    best_dnbn = max(dnbn_rows, key=lambda r: r["test_accuracy"])

    print(f"\nBest YOLOv8s:  acc={best_v8['test_accuracy']:.4f}  "
          f"f1={best_v8['test_f1_macro']:.4f}  params={best_v8['params']:,}")
    print(f"Best YOLO11s:  acc={best_11['test_accuracy']:.4f}  "
          f"f1={best_11['test_f1_macro']:.4f}  params={best_11['params']:,}")
    print(f"Best DNBN:     acc={best_dnbn['test_accuracy']:.4f}  "
          f"f1={best_dnbn['test_f1_macro']:.4f}  params={best_dnbn['total_params']:,}")

    # Save combined summary
    summary = {
        "best_yolov8s": best_v8,
        "best_yolo11s": best_11,
        "best_dnbn": best_dnbn,
        "all_yolov8s": yolov8_rows,
        "all_yolo11s": yolo11_rows,
        "all_dnbn": dnbn_rows,
    }
    with open(output_dir / "full_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to {output_dir / 'full_summary.json'}")


if __name__ == "__main__":
    main()
