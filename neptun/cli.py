"""CLI entry point for Neptun DNBN experiments."""

import argparse
import glob
import json
import os
import sys

import torch

from .config import load_sys_config
from .datasets import get_dataset
from .system import DNBNSystem
from .trainer import Trainer
from .evaluator import evaluate_system


def main():
    parser = argparse.ArgumentParser(
        description='Neptun -- Distributed Neural and Bonds Network'
    )
    parser.add_argument('--run', type=str, metavar='DATASET',
                        help='Dataset name to run (mnist, cifar10)')
    parser.add_argument('--sys-dnbn', type=str, metavar='CONFIG',
                        help='Path to system DNBN config JSON')
    parser.add_argument('--list', action='store_true',
                        help='List available experiments')
    parser.add_argument('--device', type=str, default='auto',
                        help='Compute device (cpu, cuda, mps, auto)')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.list:
        _list_experiments(repo_root)
        return

    if args.run:
        if not args.sys_dnbn:
            print("Error: --sys-dnbn <config> is required with --run")
            sys.exit(1)
        _run_experiment(args.run, args.sys_dnbn, args.device, repo_root)
        return

    parser.print_help()


def _list_experiments(repo_root):
    configs_dir = os.path.join(repo_root, 'configs')
    patterns = os.path.join(configs_dir, 'sys_dnbn_*.json')
    files = sorted(glob.glob(patterns))
    if not files:
        print("No system configs found in configs/")
        return
    print("Available experiments:")
    for path in files:
        name = os.path.basename(path)
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"  {name}  --  {data.get('name', '(unnamed)')}")


def _resolve_device(device_str):
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device_str)


def _run_experiment(dataset_name, sys_config_path, device_str, repo_root):
    device = _resolve_device(device_str)
    print(f"Device: {device}")

    if not os.path.isabs(sys_config_path):
        sys_config_path = os.path.join(repo_root, sys_config_path)

    sys_config = load_sys_config(sys_config_path, repo_root)

    print(f"\n{'=' * 60}")
    print(f"  Neptun DNBN System: {sys_config.get('name', '(unnamed)')}")
    print(f"  Dataset:  {dataset_name}")
    print(f"  Nodes:    {list(sys_config['nodes'].keys())}")
    conns = [f"{c['from']} -> {c['to']}" for c in sys_config['connections']]
    print(f"  Links:    {conns}")
    print(f"{'=' * 60}\n")

    batch_size = sys_config['training'].get('batch_size', 64)
    train_loader, val_loader, test_loader, input_shape, output_dim = get_dataset(
        dataset_name, batch_size=batch_size
    )

    system = DNBNSystem(sys_config, input_channels=input_shape[0], output_dim=output_dim)
    total_params = sum(p.numel() for p in system.parameters())
    print(f"Total parameters:      {total_params:,}")
    print(f"Communication rounds:  {system.communication_rounds}")
    print()

    print("Phase 1: Training")
    print("-" * 40)
    trainer = Trainer(system, train_loader, val_loader, sys_config)
    history = trainer.train(device)

    print(f"\nPhase 2: Evaluation")
    print("-" * 40)
    results, comm_stats = evaluate_system(system, test_loader, device)

    print(f"\n{'=' * 60}")
    print("  Final Results")
    print(f"{'=' * 60}")

    for key, stats in results.items():
        label = key if key != 'ensemble' else 'ENSEMBLE'
        print(f"\n  {label}:")
        print(f"    Accuracy:  {stats['accuracy']:.4f}  "
              f"({stats['correct']}/{stats['total']})")
        print(f"    Loss:      {stats['loss']:.4f}")

    print(f"\n  Bond Biases:")
    for conn_key, strength in system.comm.get_bond_summary().items():
        print(f"    {conn_key}: {strength:.4f}")

    print(f"\n  Communication Statistics (evaluation):")
    print(f"    Total messages routed:    {comm_stats['total_messages']}")
    print(f"    Avg message magnitude:    {comm_stats['avg_msg_magnitude']:.6f}")
    print(f"    Avg message variance:     {comm_stats['avg_msg_variance']:.6f}")
    print(f"    Max message value:        {comm_stats['max_msg_value']:.6f}")
    print(f"    Sparsity ratio:           {comm_stats['sparsity_ratio']:.6f}")

    best_val = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val) + 1
    print(f"\n  Training Summary:")
    print(f"    Best Val Accuracy:  {best_val:.4f} (epoch {best_epoch})")
    print(f"    Final Val Accuracy: {history['val_acc'][-1]:.4f}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
