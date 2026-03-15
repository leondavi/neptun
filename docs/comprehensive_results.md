# Neptun Comprehensive Results

Date: 2026-03-15

## Goal

Reduce neuron dimensions as much as possible while keeping both ensemble test Accuracy and macro F1 within 2 percentage points (absolute) of the tuned 8-node CIFAR-10 reference.

Then run a 16-node experiment using the smallest neuron setting that satisfies that constraint.

## Reference and Acceptance Threshold

Reference run (8-node tuned, 10 epochs):
- Source: [outputs/experiments/topology_sweep_tuned_20260315/all_results.csv](outputs/experiments/topology_sweep_tuned_20260315/all_results.csv)
- Config: [configs/sys_dnbn_cifar10_8node_tuned.json](configs/sys_dnbn_cifar10_8node_tuned.json)
- Ensemble Accuracy: 0.8527
- Ensemble F1: 0.8527506589889526

Acceptance thresholds (no more than 2% absolute drop):
- Minimum Accuracy: 0.8327
- Minimum F1: 0.8327506589889526

## 8-Node Neuron Sweep

Sweep outputs:
- Aggregate: [outputs/experiments/neuron_sweep_8node_20260315/all_results.csv](outputs/experiments/neuron_sweep_8node_20260315/all_results.csv)

### Summary Table

| M | C | Nodes | Params | Ensemble Acc | Ensemble F1 | Acc Delta vs Ref | F1 Delta vs Ref | Pass/Fail |
|---|---|-------|--------|--------------|-------------|------------------|-----------------|-----------|
| 128 | 128 | 8 | 3,711,921 | 0.8497 | 0.8498546481 | -0.0030 | -0.0029 | Pass |
| 96 | 96 | 8 | 2,874,673 | 0.8431 | 0.8437174559 | -0.0096 | -0.0090 | Pass |
| 64 | 64 | 8 | 2,242,225 | 0.8327 | 0.8336070180 | -0.0200 | -0.0191 | Pass |
| 48 | 48 | 8 | 2,002,801 | 0.8390 | 0.8397846222 | -0.0137 | -0.0130 | Pass |
| 32 | 32 | 8 | 1,814,577 | 0.8303 | 0.8312376738 | -0.0224 | -0.0215 | Fail |

### Per-run output files

- M128/C128: [outputs/experiments/neuron_sweep_8node_20260315/sys_dnbn_cifar10_8node_m128c128_tuned/results.csv](outputs/experiments/neuron_sweep_8node_20260315/sys_dnbn_cifar10_8node_m128c128_tuned/results.csv)
- M96/C96: [outputs/experiments/neuron_sweep_8node_20260315/sys_dnbn_cifar10_8node_m96c96_tuned/results.csv](outputs/experiments/neuron_sweep_8node_20260315/sys_dnbn_cifar10_8node_m96c96_tuned/results.csv)
- M64/C64: [outputs/experiments/neuron_sweep_8node_20260315/sys_dnbn_cifar10_8node_m64c64_tuned/results.csv](outputs/experiments/neuron_sweep_8node_20260315/sys_dnbn_cifar10_8node_m64c64_tuned/results.csv)
- M48/C48: [outputs/experiments/neuron_sweep_8node_20260315/sys_dnbn_cifar10_8node_m48c48_tuned/results.csv](outputs/experiments/neuron_sweep_8node_20260315/sys_dnbn_cifar10_8node_m48c48_tuned/results.csv)
- M32/C32: [outputs/experiments/neuron_sweep_8node_20260315/sys_dnbn_cifar10_8node_m32c32_tuned/results.csv](outputs/experiments/neuron_sweep_8node_20260315/sys_dnbn_cifar10_8node_m32c32_tuned/results.csv)

## Minimum Size Decision

Smallest tested setting that still satisfies both constraints:
- M=48, C=48

Reason:
- M=32, C=32 fails both thresholds (Accuracy 0.8303 < 0.8327 and F1 0.83124 < 0.83275).
- M=48, C=48 passes both thresholds with margin.

## 16-Node Experiment With Minimum Passing Size

Run setup:
- System config: [configs/sys_dnbn_cifar10_16node_m48c48_tuned.json](configs/sys_dnbn_cifar10_16node_m48c48_tuned.json)
- Node config: [configs/dnbn_m48_c48.json](configs/dnbn_m48_c48.json)
- Output aggregate: [outputs/experiments/neuron_sweep_16node_20260315/all_results.csv](outputs/experiments/neuron_sweep_16node_20260315/all_results.csv)
- Output run: [outputs/experiments/neuron_sweep_16node_20260315/sys_dnbn_cifar10_16node_m48c48_tuned/results.csv](outputs/experiments/neuron_sweep_16node_20260315/sys_dnbn_cifar10_16node_m48c48_tuned/results.csv)

Result:
- Nodes: 16
- Params: 3,996,321
- Ensemble Accuracy: 0.8512
- Ensemble F1: 0.8513714671
- Best Val Acc: 0.8300
- Duration: 3003.31s

Compared to 8-node reference (0.8527 Acc / 0.85275 F1):
- Accuracy delta: -0.0015
- F1 delta: -0.0014

This 16-node run remains well within the same 2% tolerance budget.

## Config Files Added For This Tuning

Node parameter files:
- [configs/dnbn_m128_c128.json](configs/dnbn_m128_c128.json)
- [configs/dnbn_m96_c96.json](configs/dnbn_m96_c96.json)
- [configs/dnbn_m64_c64.json](configs/dnbn_m64_c64.json)
- [configs/dnbn_m48_c48.json](configs/dnbn_m48_c48.json)
- [configs/dnbn_m32_c32.json](configs/dnbn_m32_c32.json)

8-node sweep system files:
- [configs/sys_dnbn_cifar10_8node_m128c128_tuned.json](configs/sys_dnbn_cifar10_8node_m128c128_tuned.json)
- [configs/sys_dnbn_cifar10_8node_m96c96_tuned.json](configs/sys_dnbn_cifar10_8node_m96c96_tuned.json)
- [configs/sys_dnbn_cifar10_8node_m64c64_tuned.json](configs/sys_dnbn_cifar10_8node_m64c64_tuned.json)
- [configs/sys_dnbn_cifar10_8node_m48c48_tuned.json](configs/sys_dnbn_cifar10_8node_m48c48_tuned.json)
- [configs/sys_dnbn_cifar10_8node_m32c32_tuned.json](configs/sys_dnbn_cifar10_8node_m32c32_tuned.json)

16-node system file:
- [configs/sys_dnbn_cifar10_16node_m48c48_tuned.json](configs/sys_dnbn_cifar10_16node_m48c48_tuned.json)

## Notes

- All runs in this sweep used 10 CIFAR-10 epochs, batch size 64, lr=0.001, cosine scheduler, communication_rounds=5, bond/comm regularization 0.0.
- The project also has a 20-epoch CIFAR benchmark in [outputs/experiments/cifar10_20ep_tuned_20260315/all_results.csv](outputs/experiments/cifar10_20ep_tuned_20260315/all_results.csv). This document keeps the threshold reference aligned to the 10-epoch tuning baseline for apples-to-apples comparison.
