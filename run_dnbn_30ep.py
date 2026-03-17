"""Run DNBN 30-epoch sweep on STL-10 (M64/C48 + M48/C48 + smaller configs)."""

import csv
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neptun.datasets import get_dataset
from neptun.evaluator import evaluate_system
from neptun.system import DNBNSystem
from neptun.trainer import Trainer


def main():
    device_str = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    output_dir = Path("outputs/experiments/stl10_yolo_vs_dnbn_20260316")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, input_shape, output_dim = get_dataset(
        "stl10", batch_size=64
    )

    repo_root = os.path.dirname(os.path.abspath(__file__))
    epochs = 30

    configs = [
        "configs/sys_dnbn_stl10_8node_m64c48_tuned.json",
        "configs/sys_dnbn_cifar10_8node_m48c48_tuned.json",
        "configs/sys_dnbn_cifar10_8node_m32c32_tuned.json",
        "configs/sys_dnbn_cifar10_8node_m24c24_tuned.json",
        "configs/sys_dnbn_cifar10_8node_m16c16_tuned.json",
    ]

    all_rows = []

    for i, config_path in enumerate(configs, 1):
        run_name = Path(config_path).stem + f"_e{epochs}"
        run_dir = output_dir / run_name
        run_dir.mkdir(exist_ok=True)

        print(f"\n{'='*72}")
        print(f"[{i}/{len(configs)}] DNBN {run_name}")
        print(f"{'='*72}")

        with open(os.path.join(repo_root, config_path)) as f:
            sys_cfg = json.load(f)

        for node_cfg in sys_cfg["nodes"].values():
            dnbn_path = node_cfg["config"]
            if not os.path.isabs(dnbn_path):
                dnbn_path = os.path.join(repo_root, dnbn_path)
            with open(dnbn_path) as f:
                node_cfg["params"] = json.load(f)

        sys_cfg["training"]["epochs"] = epochs

        system = DNBNSystem(sys_cfg, input_channels=input_shape[0],
                            output_dim=output_dim)
        total_params = sum(p.numel() for p in system.parameters())
        first_node_params = next(iter(sys_cfg["nodes"].values()))["params"]
        m_val = first_node_params.get("M", 256)
        c_val = first_node_params.get("C", 256)

        print(f"M: {m_val}, C: {c_val}, Nodes: 8, Epochs: {epochs}, Params: {total_params:,}")

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
            "M": m_val,
            "C": c_val,
            "nodes": 8,
            "total_params": total_params,
            "epochs": epochs,
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

    # Write aggregate
    agg_path = output_dir / "dnbn_sweep_30ep_results.csv"
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved aggregate: {agg_path}")

    # Summary
    print(f"\n{'='*72}")
    print("DNBN 30-Epoch Sweep Summary")
    print(f"{'='*72}")
    print(f"{'M':>4} {'C':>4} {'Accuracy':>10} {'F1':>10} {'Params':>12}")
    print("-" * 44)
    for r in all_rows:
        print(f"{r['M']:>4} {r['C']:>4} {r['test_accuracy']:>10.6f} {r['test_f1_macro']:>10.6f} {r['total_params']:>12,}")


if __name__ == "__main__":
    main()
