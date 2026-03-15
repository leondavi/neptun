# Neptun — Distributed Neural and Bonds Network (DNBN)

Neptun is a research framework for exploring **Distributed Neural and Bonds Networks (DNBN)** — a model architecture where multiple shallow neural networks cooperate through learnable communication channels and dynamic bond structures.

## Core Idea

Each DNBN instance is a shallow network with two types of neurons:

- **M task neurons** (default 512): standard hidden neurons for the classification/regression task.
- **C communication neurons** (default 128): dedicated neurons that send and receive messages to/from other DNBN instances in the system.

Multiple DNBN instances are connected through a learnable bond matrix. The bonds control:
- **Who** communicates with whom (connectivity graph).
- **What** information is transmitted (learned message content).
- **How strongly** each connection operates (bond strength, trained via gradient descent with sparsity pressure).

Bandwidth allocation determines how communication neuron capacity is divided among connected models, with support for overlapping allocations where multiple models can send to shared receiving neurons.

## Quick Start

```bash
# Install environment and download datasets
./Neptun.sh --install

# List available experiments
./Neptun.sh --list

# Run MNIST experiment with a 2-node DNBN system
./Neptun.sh --run mnist --sys-dnbn configs/sys_dnbn_mnist_2node.json

# Run CIFAR-10 experiment
./Neptun.sh --run cifar10 --sys-dnbn configs/sys_dnbn_cifar10_2node.json

# Clean up
./Neptun.sh --clean
```

## Project Structure

```
neptun/                  # Python package
  model.py               # DNBN single-instance model
  communication.py       # Inter-model communication layer
  system.py              # System of connected DNBN instances
  trainer.py             # Training loop (Phase 1)
  evaluator.py           # Prediction & evaluation (Phase 2)
  datasets.py            # Dataset loading (MNIST, CIFAR-10)
  config.py              # Configuration parsing
  cli.py                 # CLI entry point
configs/                 # Configuration files
  dnbn_default.json      # Default DNBN model parameters
  sys_dnbn_*.json        # System topology configs
docs/                    # Detailed documentation
agent/                   # Agent background and plans
tmp/                     # Downloaded datasets (gitignored)
Neptun.sh                # Main handler script
```

## Configuration

### System DNBN Config (`sys_dnbn_*.json`)

Defines the full system: which models exist, their connectivity, communication rates, and bandwidth allocation. Each node references a DNBN config file.

### DNBN Config (`dnbn_*.json`)

Defines per-model parameters: number of task neurons (M), communication neurons (C), activation function, and dropout.

See [docs/architecture.md](docs/architecture.md) for full details.

## Phases

1. **Training**: All DNBN instances in the system train cooperatively. Each processes the input, exchanges messages through communication neurons over multiple rounds, and backpropagates through the entire communication graph including bond strengths.

2. **Prediction & Evaluation**: The trained system is evaluated on held-out test data. Per-node accuracy, ensemble accuracy (averaged logits), and bond analysis are reported.

## Theory

See [docs/dnbn_theory.md](docs/dnbn_theory.md) and [agent/background.md](agent/background.md) for the theoretical foundations of DNBN including bond formation, communication protocols, and training objectives.

## License

Research use only.
