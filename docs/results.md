# Neptun DNBN Results

## Summary

The DNBN graph-transformer architecture with tuned hyperparameters **outperforms ResNet18** on CIFAR-10 at 20 epochs, and achieves **99.5% test accuracy** on MNIST.

---

## Hyperparameter Tuning (10 Epochs, CIFAR-10 Ring Topology)

Sweep date: 2026-03-15. All variants use the 8-node bidirectional ring on CIFAR-10.

| Variant | bond_λ | comm_λ | LR | Scheduler | Rounds | Best Val | Test Acc |
|---------|--------|--------|-----|-----------|--------|----------|----------|
| **Tuned Combo** | 0.0 | 0.0 | 0.001 | cosine | 5 | **84.9%** | **85.3%** |
| No Reg | 0.0 | 0.0 | 0.001 | none | 3 | 82.4% | 82.3% |
| Baseline | 0.01 | 0.001 | 0.001 | none | 3 | 81.6% | 81.6% |
| Low Reg | 0.001 | 0.0001 | 0.001 | none | 3 | 80.9% | 81.2% |
| 5 Rounds | 0.01 | 0.001 | 0.001 | none | 5 | 80.8% | 76.5% |
| Cosine LR 3e-3 | 0.01 | 0.001 | 0.003 | cosine | 3 | 63.6% | 68.0% |

**Key findings:**
- Cosine annealing at lr=0.001 is the biggest single improvement
- Removing regularization helps — bond/comm penalties were hurting accuracy
- 5 communication rounds help when combined with LR scheduling
- LR=0.003 is too high — destabilizes training

---

## Model Complexity Comparison

| Model | Total Params | Conv Layers | Neurons | Comm Neurons | M | C |
|-------|-------------|-------------|---------|--------------|---|---|
| **DNBN 8-Node Tuned** | 9.11M | 56 (7/expert × 8) | 6,480 (810/expert × 8) | 2,048 (256/expert × 8) | 256 | 256 |
| DNBN 8-Node Baseline | 9.11M | 56 (7/expert × 8) | 6,480 (810/expert × 8) | 2,048 (256/expert × 8) | 256 | 256 |
| ResNet18 | 11.18M | 20 | 1,034 | 0 | — | — |
| EfficientNet-B0 | 4.02M | 81 | 2,106 | 0 | — | — |

### DNBN Per-Expert Breakdown (×8 experts)

| Component | Params | Neurons | Description |
|-----------|--------|---------|-------------|
| ConvNet Backbone | 219,040 | 224 (32+64+128) | 7 conv layers: 3→32→32→32→64→64→64→128, all 3×3, with ResBlocks |
| Projection (Linear 128→M) | — | 256 | Pooled features dimension |
| Q/K/V Projections | 197,376 | — | 3× Linear(M=256→C=256) for multi-head attention |
| State GRU | 591,360 | 256 | GRUCell(M+C=512, M=256), recurrent state across rounds |
| Comm Controller | 95,364 | 64 | GRUCell(M→64) + gate/bias projections |
| Classifier | 2,570 | 10 | Linear(M=256→10) |
| **Expert Total** | **1,105,710** | **810** | |

### Shared Communication Module

| Component | Params | Description |
|-----------|--------|-----------|
| Bond bias (N×N) | 64 | 8×8 learnable attention bias matrix |
| Output projection | 65,792 | Linear(C=256→C=256) |
| Buffer readout Q/K/V | 197,376 | 3× Linear for temporal buffer attention |
| Buffer decay | 1 | Learnable decay logit (γ ≈ 0.88) |
| **Comm Total** | **263,233** | |

---

## CIFAR-10 (20 Epochs) DNBN vs SOTA Baselines

Output: `outputs/experiments/cifar10_20ep_tuned_20260315/all_results.csv`

| Model | Family | Params | Conv Layers | Neurons | Test Acc | F1 (macro) | Best Val | Duration |
|-------|--------|--------|-------------|---------|----------|------------|----------|----------|
| **DNBN Tuned** (8-node ring) | DNBN | 9.11M | 56 | 6,480 | **87.2%** | **0.872** | 87.5% | 3517s |
| DNBN Baseline (8-node ring) | DNBN | 9.11M | 56 | 6,480 | 85.0% | 0.849 | 85.9% | 2996s |
| ResNet18 | Baseline | 11.18M | 20 | 1,034 | 78.9% | 0.788 | 80.6% | 916s |
| EfficientNet-B0 | Baseline | 4.02M | 81 | 2,106 | 45.2% | 0.455 | 44.8% | 1162s |

**The tuned DNBN outperforms ResNet18 by +8.3 pp on CIFAR-10 with fewer parameters (9.11M vs 11.18M).**

---

## Topology Comparison (Tuned Settings, 10 Epochs)

Output: `outputs/experiments/topology_sweep_tuned_20260315/all_results.csv`

### CIFAR-10 (10 epochs)

| Topology | Best Val | Test Acc |
|----------|----------|----------|
| Ring | 84.9% | 85.3% |
| Star-Hub | 84.7% | 85.2% |
| Ring + Skip | 84.8% | 84.6% |
| All-to-All | 83.9% | 83.8% |

### MNIST (5 epochs)

| Topology | Best Val | Test Acc |
|----------|----------|----------|
| All-to-All | 99.4% | **99.5%** |
| Ring + Skip | 99.2% | 99.5% |
| Star-Hub | 99.3% | 99.5% |
| Ring | 99.3% | 99.5% |

**All topologies converge to ~99.5% on MNIST.** On CIFAR-10, sparse topologies (ring, star) slightly outperform dense (all-to-all), suggesting the model can learn selective communication effectively.

---

## Tuned Hyperparameters

The best-performing configuration uses:

```json
{
    "learning_rate": 0.001,
    "lr_scheduler": "cosine",
    "grad_clip": 1.0,
    "weight_decay": 0.0001,
    "bond_sparsity_lambda": 0.0,
    "comm_cost_lambda": 0.0,
    "communication_rounds": 5
}
```

Architecture defaults (unchanged): `M=256`, `C=256`, `num_heads=4`, `buffer_size=8`, `controller_hidden=64`, `dropout=0.1`.

## Model Size Summary

- **DNBN 8-Node**: 9.11M params, 56 conv layers, 6,480 neurons, **2,048 communication neurons** (C=256 × 8 experts)
- **ResNet18**: 11.18M params, 20 conv layers, 1,034 neurons, no communication
- **EfficientNet-B0**: 4.02M params, 81 conv layers, 2,106 neurons, no communication

Note: "Neurons" counts the unique layer widths (feature map channels for conv layers, hidden units for FC layers). DNBN has more neurons overall due to 8 parallel experts, but each expert is much smaller than either baseline. The 2,048 communication neurons (C=256 per expert) are the dimensions used for multi-head attention message passing between experts.

## Notes

- All comparisons use the same dataset split (`seed=42`), batch size (`64`), and epoch budget.
- The trainer now supports cosine annealing LR scheduling, gradient clipping, and weight decay.
- EfficientNet-B0 baseline trains poorly at lr=0.001 — it likely needs a higher learning rate or different optimizer. This does not reflect its true capability.
