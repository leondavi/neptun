"""Run DNBN neuron sweep for YOLO comparison.

Runs M/C sizes: 48, 32, 24, 16 on 8-node CIFAR-10 with 10 epochs.
Saves results to outputs/experiments/yolo_vs_dnbn_20260316/
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neptun.config import load_sys_config
from neptun.datasets import get_dataset
from neptun.evaluator import evaluate_system
from neptun.system import DNBNSystem
from neptun.trainer import Trainer


def main():
    device_str = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    output_dir = Path("outputs/experiments/yolo_vs_dnbn_20260316")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, input_shape, output_dim = get_dataset(
        "cifar10", batch_size=64
    )

    configs = [
        "configs/sys_dnbn_cifar10_8node_m48c48_tuned.json",
        "configs/sys_dnbn_cifar10_8node_m32c32_tuned.json",
        "configs/sys_dnbn_cifar10_8node_m24c24_tuned.json",
        "configs/sys_dnbn_cifar10_8node_m16c16_tuned.json",
    ]

    all_rows = []

    for i, config_path in enumerate(configs, 1):
        run_name = Path(config_path).stem
        run_dir = output_dir / run_name
        run_dir.mkdir(exist_ok=True)

        print(f"\n{'='*72}")
        print(f"[{i}/{len(configs)}] {run_name}")
        print(f"{'='*72}")

        sys_cfg = json.loads(json.dumps(
            json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)))
        ))

        # Resolve node configs
        repo_root = os.path.dirname(os.path.abspath(__file__))
        for node_id, node_cfg in sys_cfg["nodes"].items():
            dnbn_path = node_cfg["config"]
            if not os.path.isabs(dnbn_path):
                dnbn_path = os.path.join(repo_root, dnbn_path)
            with open(dnbn_path) as f:
                node_cfg["params"] = json.load(f)

        sys_cfg["training"]["epochs"] = 10

        system = DNBNSystem(sys_cfg, input_channels=input_shape[0], output_dim=output_dim)
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
        print(f"\nResults: acc={ensemble['accuracy']:.6f} f1={ensemble['f1_macro']:.6f} params={total_params:,} time={elapsed:.1f}s")

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

        csv_path = run_dir / "results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        print(f"Saved: {csv_path}")

    # Write aggregate
    agg_path = output_dir / "dnbn_sweep_results.csv"
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved aggregate: {agg_path}")

    # Print summary
    print(f"\n{'='*72}")
    print("DNBN Sweep Summary")
    print(f"{'='*72}")
    print(f"{'M/C':>6} {'Accuracy':>10} {'F1':>10} {'Params':>12}")
    print("-" * 42)
    for r in all_rows:
        print(f"{r['M_C']:>6} {r['test_accuracy']:>10.6f} {r['test_f1_macro']:>10.6f} {r['total_params']:>12,}")


if __name__ == "__main__":
    main()
