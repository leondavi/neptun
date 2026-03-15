"""Batch experiment runner for DNBN system configs.

Runs multiple system configs, writes per-experiment text/csv results,
and writes aggregate CSV summaries.
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import os
from datetime import datetime, timezone
from typing import Dict, List

import torch

from .config import load_sys_config
from .datasets import get_dataset
from .evaluator import evaluate_system
from .system import DNBNSystem
from .trainer import Trainer


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _infer_dataset(config_path: str) -> str:
    base = os.path.basename(config_path).lower()
    if "mnist" in base:
        return "mnist"
    if "cifar10" in base or "cifar" in base:
        return "cifar10"
    raise ValueError(
        f"Unable to infer dataset from config filename: {config_path}. "
        "Expected 'mnist' or 'cifar10' in filename."
    )


def _safe_json(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_experiment_text(
    output_path: str,
    dataset: str,
    config_path: str,
    sys_config: Dict,
    history: Dict,
    eval_results: Dict,
    comm_stats: Dict,
    bond_summary: Dict,
    duration_sec: float,
) -> None:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append(f"Experiment: {sys_config.get('name', '(unnamed)')}")
    lines.append(f"Config:     {config_path}")
    lines.append(f"Dataset:    {dataset}")
    lines.append(f"Started:    {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Duration:   {duration_sec:.2f}s")
    lines.append("=" * 72)
    lines.append("")

    lines.append("Training")
    lines.append("-" * 72)
    best_val = max(history["val_acc"])
    best_epoch = history["val_acc"].index(best_val) + 1
    lines.append(f"Epochs:             {len(history['val_acc'])}")
    lines.append(f"Best validation:    {best_val:.6f} (epoch {best_epoch})")
    lines.append(f"Final validation:   {history['val_acc'][-1]:.6f}")
    lines.append("")

    lines.append("Evaluation")
    lines.append("-" * 72)
    for key, stats in eval_results.items():
        label = key if key != "ensemble" else "ENSEMBLE"
        lines.append(f"{label}:")
        lines.append(f"  accuracy: {stats['accuracy']:.6f}")
        lines.append(f"  loss:     {stats['loss']:.6f}")
        lines.append(f"  precision_macro: {stats['precision_macro']:.6f}")
        lines.append(f"  f1_macro:        {stats['f1_macro']:.6f}")
        lines.append(f"  correct:  {stats['correct']} / {stats['total']}")
    lines.append("")

    lines.append("Communication Statistics")
    lines.append("-" * 72)
    lines.append(f"Total messages routed:    {comm_stats['total_messages']}")
    lines.append(f"Messages skipped (rate):  {comm_stats['skipped_by_rate']}")
    lines.append(f"Avg message magnitude:    {comm_stats['avg_msg_magnitude']:.6f}")
    lines.append(f"Avg message variance:     {comm_stats['avg_msg_variance']:.6f}")
    lines.append(f"Max message value:        {comm_stats['max_msg_value']:.6f}")
    lines.append("")
    lines.append("Per-connection communication:")
    for conn_key, cs in comm_stats["per_connection"].items():
        lines.append(
            f"  {conn_key}: messages={cs['messages']} bandwidth={cs['bandwidth']} "
            f"avg_mag={cs['avg_magnitude']:.6f} avg_var={cs['avg_variance']:.6f} "
            f"max={cs['max_value']:.6f}"
        )
    lines.append("")

    lines.append("Bond strengths")
    lines.append("-" * 72)
    for conn_key, strength in bond_summary.items():
        lines.append(f"{conn_key}: {strength:.6f}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_single_csv(output_path: str, row: Dict) -> None:
    row = {k: _safe_json(v) for k, v in row.items()}
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_aggregate_csv(output_path: str, rows: List[Dict]) -> None:
    if not rows:
        return

    all_keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                all_keys.append(key)
                seen.add(key)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            safe = {k: _safe_json(v) for k, v in row.items()}
            writer.writerow(safe)


def run_batch(
    output_dir: str,
    config_glob: str,
    mnist_epochs_override: int | None,
    cifar_epochs_override: int | None,
    device_str: str,
) -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(output_dir, exist_ok=True)

    config_paths = sorted(glob.glob(os.path.join(repo_root, config_glob)))
    if not config_paths:
        print(f"No config files matched: {config_glob}")
        return 1

    device = _resolve_device(device_str)
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Experiments found: {len(config_paths)}")

    summary_rows: List[Dict] = []
    error_rows: List[Dict] = []

    for index, config_path in enumerate(config_paths, start=1):
        dataset = _infer_dataset(config_path)
        config_name = os.path.basename(config_path)
        run_name = os.path.splitext(config_name)[0]
        run_dir = os.path.join(output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        print("\n" + "=" * 72)
        print(f"[{index}/{len(config_paths)}] Running {config_name} ({dataset})")
        print("=" * 72)

        start = datetime.now(timezone.utc)
        try:
            sys_cfg = load_sys_config(config_path, repo_root)
            sys_cfg = copy.deepcopy(sys_cfg)

            if dataset == "mnist" and mnist_epochs_override is not None:
                sys_cfg["training"]["epochs"] = mnist_epochs_override
            if dataset == "cifar10" and cifar_epochs_override is not None:
                sys_cfg["training"]["epochs"] = cifar_epochs_override

            batch_size = sys_cfg["training"].get("batch_size", 64)
            train_loader, val_loader, test_loader, input_shape, output_dim = get_dataset(
                dataset, batch_size=batch_size
            )

            system = DNBNSystem(sys_cfg, input_channels=input_shape[0], output_dim=output_dim)
            total_params = sum(p.numel() for p in system.parameters())
            print(f"Nodes: {len(system.nodes)} | Params: {total_params:,}")
            print(f"Epochs: {sys_cfg['training'].get('epochs', 0)}")

            trainer = Trainer(system, train_loader, val_loader, sys_cfg)
            history = trainer.train(device)
            eval_results, comm_stats = evaluate_system(system, test_loader, device)
            bond_summary = system.comm.get_bond_summary()

            duration_sec = (datetime.now(timezone.utc) - start).total_seconds()
            ensemble = eval_results["ensemble"]
            best_val = max(history["val_acc"])
            final_val = history["val_acc"][-1]
            first_node_cfg = next(iter(sys_cfg["nodes"].values()))["params"]

            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "config_name": config_name,
                "run_name": run_name,
                "dataset": dataset,
                "system_name": sys_cfg.get("name", "(unnamed)"),
                "communication_neurons_size": first_node_cfg.get("C", 128),
                "queue_size": first_node_cfg.get("queue_size", 3),
                "epochs": sys_cfg["training"].get("epochs"),
                "batch_size": sys_cfg["training"].get("batch_size"),
                "learning_rate": sys_cfg["training"].get("learning_rate"),
                "communication_rounds": sys_cfg["training"].get("communication_rounds"),
                "bond_sparsity_lambda": sys_cfg["training"].get("bond_sparsity_lambda"),
                "nodes": len(system.nodes),
                "connections": len(sys_cfg.get("connections", [])),
                "total_params": total_params,
                "best_val_acc": best_val,
                "final_val_acc": final_val,
                "ensemble_test_acc": ensemble["accuracy"],
                "ensemble_test_loss": ensemble["loss"],
                "ensemble_precision_macro": ensemble["precision_macro"],
                "ensemble_f1_macro": ensemble["f1_macro"],
                "ensemble_correct": ensemble["correct"],
                "ensemble_total": ensemble["total"],
                "comm_total_messages": comm_stats["total_messages"],
                "comm_skipped_by_rate": comm_stats["skipped_by_rate"],
                "comm_avg_msg_magnitude": comm_stats["avg_msg_magnitude"],
                "comm_avg_msg_variance": comm_stats["avg_msg_variance"],
                "comm_max_msg_value": comm_stats["max_msg_value"],
                "duration_sec": duration_sec,
                "per_node_metrics": eval_results,
                "per_connection_comm": comm_stats["per_connection"],
                "bond_summary": bond_summary,
            }

            txt_path = os.path.join(run_dir, "results.txt")
            csv_path = os.path.join(run_dir, "results.csv")
            _write_experiment_text(
                txt_path,
                dataset,
                config_path,
                sys_cfg,
                history,
                eval_results,
                comm_stats,
                bond_summary,
                duration_sec,
            )
            _write_single_csv(csv_path, row)
            summary_rows.append(row)
            print(f"Saved: {txt_path}")
            print(f"Saved: {csv_path}")

        except Exception as exc:  # noqa: BLE001
            duration_sec = (datetime.now(timezone.utc) - start).total_seconds()
            err = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "config_name": config_name,
                "dataset": dataset,
                "duration_sec": duration_sec,
                "error": repr(exc),
            }
            error_rows.append(err)
            print(f"FAILED: {config_name} -> {exc}")

    all_csv = os.path.join(output_dir, "all_results.csv")
    _write_aggregate_csv(all_csv, summary_rows)
    print(f"\nSaved aggregate CSV: {all_csv}")

    if error_rows:
        err_csv = os.path.join(output_dir, "errors.csv")
        _write_aggregate_csv(err_csv, error_rows)
        print(f"Saved error CSV: {err_csv}")

    print(
        f"Completed {len(summary_rows)} / {len(config_paths)} experiments "
        f"successfully."
    )
    return 0 if not error_rows else 2


def main():
    parser = argparse.ArgumentParser(description="Batch runner for DNBN experiments")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for experiment outputs",
    )
    parser.add_argument(
        "--config-glob",
        default="configs/sys_dnbn_*8node*.json",
        help="Glob pattern (repo-relative) for system config files",
    )
    parser.add_argument(
        "--mnist-epochs-override",
        type=int,
        default=5,
        help="Optional epoch override for MNIST runs",
    )
    parser.add_argument(
        "--cifar-epochs-override",
        type=int,
        default=10,
        help="Optional epoch override for CIFAR-10 runs",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, cuda, mps",
    )
    args = parser.parse_args()

    raise SystemExit(
        run_batch(
            output_dir=args.output_dir,
            config_glob=args.config_glob,
            mnist_epochs_override=args.mnist_epochs_override,
            cifar_epochs_override=args.cifar_epochs_override,
            device_str=args.device,
        )
    )


if __name__ == "__main__":
    main()
