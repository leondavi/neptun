# Neptun — Distributed Neural and Bonds Network (DNBN)

Neptun is a research framework for **Distributed Neural and Bonds Networks (DNBN)** — a graph-transformer architecture where multiple expert nodes with ConvNet backbones cooperate through multi-head attention message passing, recurrent state evolution, and learnable topology.

## Key Results

| Model | Params | Test Acc (CIFAR-10, 20ep) | Test Acc (MNIST, 5ep) |
|-------|--------|---------------------------|----------------------|
| **DNBN 8-Node Tuned** | 9.11M | **87.2%** | **99.5%** |
| ResNet18 | 11.18M | 78.9% | — |
| EfficientNet-B0 | 4.02M | 45.2%* | — |

\* EfficientNet-B0 undertrained — needs different optimizer/LR. See [docs/results.md](docs/results.md) for full results, hyperparameter tuning, topology comparisons, and model complexity analysis.

## Architecture

Each DNBN system consists of **N expert nodes** connected via a communication graph. Each expert has:

- **ConvNet Backbone**: 7-layer ResNet-like feature extractor (3→32→64→128) producing M-dimensional embeddings
- **Communication Controller**: GRU-based controller that learns send/receive gates and attention biases
- **Multi-Head Attention**: Q/K/V projections for graph transformer message passing (4 heads, C=256)
- **Recurrent State GRU**: Hidden state evolves across communication rounds via BPTT
- **Temporal Buffers**: Sliding window of past messages with learned decay for historical context

Communication rounds are treated as **recurrent time steps** — shared parameters, evolving states, trained via backpropagation through time. This enables iterative reasoning rather than single-pass feedforward processing.

Supported topologies: **ring**, **all-to-all**, **ring + skip connections**, **star-hub**.

See [docs/architecture.md](docs/architecture.md) for the full architecture specification.

## Quick Start

```bash
# Install environment and download datasets
./Neptun.sh --install

# List available experiments
./Neptun.sh --list

# Run MNIST experiment with an 8-node DNBN system
./Neptun.sh --run mnist --sys-dnbn configs/sys_dnbn_mnist_8node_tuned.json

# Run CIFAR-10 experiment
./Neptun.sh --run cifar10 --sys-dnbn configs/sys_dnbn_cifar10_8node_tuned.json

# Run CIFAR-10 comparison against baselines (ResNet18, EfficientNet-B0)
./.venv/bin/python -m neptun.cifar_compare --epochs 20

# Run full experiment sweep across topologies
./.venv/bin/python -m neptun.experiment_runner --config-glob 'configs/sys_dnbn_*8node*_tuned.json'

# Clean up
./Neptun.sh --clean
```

## Project Structure

```
neptun/                  # Python package
  model.py               # Expert node: ConvNet backbone, GRU controller, state GRU
  communication.py       # Graph transformer: multi-head attention, temporal buffers
  system.py              # System orchestrator: recurrent communication rounds
  trainer.py             # Training loop with cosine LR, grad clipping, weight decay
  evaluator.py           # Prediction & evaluation
  baselines.py           # ResNet18, EfficientNet-B0 baselines
  cifar_compare.py       # DNBN vs baselines comparison script
  experiment_runner.py   # Automated experiment sweep runner
  datasets.py            # Dataset loading (MNIST, CIFAR-10)
  config.py              # Configuration parsing
  cli.py                 # CLI entry point
configs/                 # Configuration files
  dnbn_default.json      # Default DNBN model parameters (M=256, C=256, heads=4)
  sys_dnbn_*_tuned.json  # Tuned system configs for each topology
  sys_dnbn_*.json        # Base system topology configs
docs/                    # Documentation
  architecture.md        # Full architecture specification
  results.md             # Experiment results and analysis
  dnbn_theory.md         # Theoretical foundations
  getting_started.md     # Setup guide
Neptun.sh                # Main handler script
```

## Configuration

### System Config (`sys_dnbn_*.json`)

Defines the system: number of nodes, topology (adjacency matrix), communication rounds, regularization lambdas, and training settings (LR, scheduler, gradient clipping, weight decay).

### DNBN Config (`dnbn_default.json`)

Defines per-expert parameters: feature dimension (M=256), communication dimension (C=256), number of attention heads (4), temporal buffer size (8), controller hidden size (64), and dropout.

## Training

All expert nodes train cooperatively end-to-end:

1. Each expert extracts features from the shared input via its ConvNet backbone
2. Experts exchange messages through multi-head attention over multiple communication rounds
3. Hidden states evolve recurrently across rounds
4. Each expert produces a classification; the ensemble averages logits
5. Gradients flow through the entire communication graph via BPTT

The trainer supports **cosine annealing LR scheduling**, **gradient clipping**, and **weight decay**.

## Theory

See [docs/dnbn_theory.md](docs/dnbn_theory.md) and [agent/background.md](agent/background.md) for the theoretical foundations — bond formation, communication protocols, and training objectives.

## Citation

If this repository is useful in your research, please cite it using [CITATION.cff](CITATION.cff).

Author profile:
- David Leon: https://scholar.google.com/citations?user=J3YceFQAAAAJ&hl=en

## License

Research use only.
