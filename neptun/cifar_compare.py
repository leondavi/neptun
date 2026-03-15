"""CIFAR-10 20-epoch comparison: DNBN vs two CNN baselines.

Runs two DNBN system configs and two baseline CNNs (ResNet18, DenseNet121)
with the same dataset split seed and writes TXT/CSV outputs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from datetime import datetime, timezone

import torch

from .baselines import build_baseline, evaluate_baseline, train_baseline
from .config import load_sys_config
from .datasets import get_dataset
from .evaluator import evaluate_system
from .system import DNBNSystem
from .trainer import Trainer


def _resolve_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device_str)


def _safe_json(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_row_csv(path, row):
    row = {k: _safe_json(v) for k, v in row.items()}
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_aggregate_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _safe_json(v) for k, v in row.items()})


def _write_text(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def run_comparison(output_dir, dnbn_configs, epochs, batch_size, lr, seed, device_str):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = _resolve_device(device_str)

    os.makedirs(output_dir, exist_ok=True)

    rows = []

    train_loader, val_loader, test_loader, input_shape, output_dim = get_dataset(
        'cifar10', batch_size=batch_size, seed=seed
    )

    # DNBN runs
    for config_rel in dnbn_configs:
        config_path = config_rel
        if not os.path.isabs(config_path):
            config_path = os.path.join(repo_root, config_rel)

        cfg = copy.deepcopy(load_sys_config(config_path, repo_root))
        cfg['training']['epochs'] = epochs
        cfg['training']['batch_size'] = batch_size
        cfg['training']['learning_rate'] = lr

        run_name = os.path.splitext(os.path.basename(config_path))[0]
        run_dir = os.path.join(output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n[DNBN] {run_name} | epochs={epochs}")
        start = datetime.now(timezone.utc)

        system = DNBNSystem(cfg, input_channels=input_shape[0], output_dim=output_dim)
        trainer = Trainer(system, train_loader, val_loader, cfg)
        history = trainer.train(device)
        results, comm_stats = evaluate_system(system, test_loader, device)
        duration = (datetime.now(timezone.utc) - start).total_seconds()

        ensemble = results['ensemble']
        best_val = max(history['val_acc'])

        row = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'model_family': 'dnbn',
            'model_name': run_name,
            'dataset': 'cifar10',
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'seed': seed,
            'communication_neurons_size': next(iter(cfg['nodes'].values()))['params'].get('C', 128),
            'queue_size': next(iter(cfg['nodes'].values()))['params'].get('queue_size', 3),
            'ensemble_test_acc': ensemble['accuracy'],
            'ensemble_test_loss': ensemble['loss'],
            'ensemble_precision_macro': ensemble['precision_macro'],
            'ensemble_f1_macro': ensemble['f1_macro'],
            'best_val_acc': best_val,
            'final_val_acc': history['val_acc'][-1],
            'comm_total_messages': comm_stats['total_messages'],
            'comm_avg_msg_magnitude': comm_stats['avg_msg_magnitude'],
            'comm_avg_msg_variance': comm_stats['avg_msg_variance'],
            'duration_sec': duration,
            'per_node_metrics': results,
            'per_connection_comm': comm_stats['per_connection'],
            'bond_summary': system.comm.get_bond_summary(),
        }

        txt_lines = [
            '=' * 72,
            f"DNBN Experiment: {run_name}",
            f"Config: {config_path}",
            f"Dataset: CIFAR-10",
            f"Epochs: {epochs}",
            f"Seed: {seed}",
            '=' * 72,
            '',
            f"Ensemble Accuracy:  {ensemble['accuracy']:.6f}",
            f"Ensemble Loss:      {ensemble['loss']:.6f}",
            f"Ensemble Precision: {ensemble['precision_macro']:.6f}",
            f"Ensemble F1:        {ensemble['f1_macro']:.6f}",
            f"Best Val Acc:       {best_val:.6f}",
            f"Final Val Acc:      {history['val_acc'][-1]:.6f}",
            '',
            f"Communication messages: {comm_stats['total_messages']}",
            f"Avg msg magnitude:      {comm_stats['avg_msg_magnitude']:.6f}",
            f"Avg msg variance:       {comm_stats['avg_msg_variance']:.6f}",
            f"Duration (s):           {duration:.2f}",
        ]

        _write_text(os.path.join(run_dir, 'results.txt'), txt_lines)
        _write_row_csv(os.path.join(run_dir, 'results.csv'), row)
        rows.append(row)

    # Baseline runs
    for baseline_name in ['resnet18', 'efficientnet_b0']:
        run_name = f'baseline_{baseline_name}'
        run_dir = os.path.join(output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n[BASELINE] {baseline_name} | epochs={epochs}")
        start = datetime.now(timezone.utc)

        model = build_baseline(baseline_name, output_dim)
        history = train_baseline(
            model, train_loader, val_loader,
            epochs=epochs, lr=lr, device=device
        )
        metrics = evaluate_baseline(model, test_loader, device=device)
        duration = (datetime.now(timezone.utc) - start).total_seconds()

        row = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'model_family': 'baseline',
            'model_name': baseline_name,
            'dataset': 'cifar10',
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'seed': seed,
            'communication_neurons_size': None,
            'queue_size': None,
            'ensemble_test_acc': metrics['accuracy'],
            'ensemble_test_loss': metrics['loss'],
            'ensemble_precision_macro': metrics['precision_macro'],
            'ensemble_f1_macro': metrics['f1_macro'],
            'best_val_acc': max(history['val_acc']),
            'final_val_acc': history['val_acc'][-1],
            'comm_total_messages': 0,
            'comm_avg_msg_magnitude': 0.0,
            'comm_avg_msg_variance': 0.0,
            'duration_sec': duration,
        }

        txt_lines = [
            '=' * 72,
            f"Baseline Experiment: {baseline_name}",
            f"Dataset: CIFAR-10",
            f"Epochs: {epochs}",
            f"Seed: {seed}",
            '=' * 72,
            '',
            f"Accuracy:  {metrics['accuracy']:.6f}",
            f"Loss:      {metrics['loss']:.6f}",
            f"Precision: {metrics['precision_macro']:.6f}",
            f"F1:        {metrics['f1_macro']:.6f}",
            f"Best Val:  {max(history['val_acc']):.6f}",
            f"Final Val: {history['val_acc'][-1]:.6f}",
            f"Duration (s): {duration:.2f}",
        ]

        _write_text(os.path.join(run_dir, 'results.txt'), txt_lines)
        _write_row_csv(os.path.join(run_dir, 'results.csv'), row)
        rows.append(row)

    _write_aggregate_csv(os.path.join(output_dir, 'all_results.csv'), rows)
    print(f"\nSaved aggregate CSV: {os.path.join(output_dir, 'all_results.csv')}")


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 DNBN vs SOTA baseline runner')
    parser.add_argument(
        '--output-dir', required=True,
        help='Output directory for comparison artifacts'
    )
    parser.add_argument(
        '--dnbn-configs', nargs='+',
        default=[
            'configs/sys_dnbn_cifar10_8node.json',
            'configs/sys_dnbn_cifar10_8node_all2all.json',
        ],
        help='DNBN config files to compare (default: two CIFAR-10 DNBN models)'
    )
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    run_comparison(
        output_dir=args.output_dir,
        dnbn_configs=args.dnbn_configs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        seed=args.seed,
        device_str=args.device,
    )


if __name__ == '__main__':
    main()
