# Getting Started with Neptun

## Prerequisites

- Python 3.9 or later
- macOS, Linux, or Windows with WSL
- ~500MB disk space for datasets

## Installation

```bash
# Clone the repository
cd /path/to/neptun

# Install virtual environment, dependencies, and download datasets
./Neptun.sh --install
```

This creates a `.venv` virtual environment with PyTorch and downloads MNIST and CIFAR-10 datasets to the `tmp/` directory.

## Running Your First Experiment

### List Available Experiments

```bash
./Neptun.sh --list
```

This shows all system DNBN configurations in the `configs/` directory.

### Run MNIST Demo

```bash
./Neptun.sh --run mnist --sys-dnbn configs/sys_dnbn_mnist_2node.json
```

This runs a 2-node DNBN system on MNIST:
- **Phase 1 (Training)**: Both DNBN instances train cooperatively for 10 epochs, exchanging messages through 3 communication rounds per forward pass.
- **Phase 2 (Evaluation)**: The trained system is evaluated on the MNIST test set. Per-node accuracy, ensemble accuracy, and bond analysis are printed.

### Run CIFAR-10 Demo

```bash
./Neptun.sh --run cifar10 --sys-dnbn configs/sys_dnbn_cifar10_2node.json
```

CIFAR-10 is harder, so this configuration uses 20 epochs.

## Understanding the Output

After training, the system reports:

1. **Per-node accuracy**: How well each individual DNBN instance performs
2. **Ensemble accuracy**: Accuracy when averaging logits across all nodes (typically higher)
3. **Bond analysis**: The learned bond strength for each connection — values near 0 indicate prunable connections, values near 1 indicate critical communication links
4. **Training history**: Best and final validation accuracy

## Creating Custom Experiments

### 1. Create a DNBN Config

Create a file `configs/dnbn_custom.json`:

```json
{
    "M": 256,
    "C": 64,
    "activation": "relu",
    "dropout": 0.2
}
```

### 2. Create a System Config

Create a file `configs/sys_dnbn_custom.json`:

```json
{
    "name": "Custom 3-Node System",
    "nodes": {
        "dnbn_0": {"config": "configs/dnbn_custom.json"},
        "dnbn_1": {"config": "configs/dnbn_custom.json"},
        "dnbn_2": {"config": "configs/dnbn_custom.json"}
    },
    "connections": [
        {"from": "dnbn_0", "to": "dnbn_1", "rate": 1.0, "send_bandwidth": 32, "send_offset": 0, "recv_offset": 0},
        {"from": "dnbn_1", "to": "dnbn_2", "rate": 1.0, "send_bandwidth": 32, "send_offset": 0, "recv_offset": 0},
        {"from": "dnbn_2", "to": "dnbn_0", "rate": 1.0, "send_bandwidth": 32, "send_offset": 0, "recv_offset": 0}
    ],
    "training": {
        "epochs": 15,
        "batch_size": 64,
        "learning_rate": 0.001,
        "communication_rounds": 4,
        "bond_sparsity_lambda": 0.01
    },
    "evaluation": {
        "batch_size": 256
    }
}
```

### 3. Run It

```bash
./Neptun.sh --run mnist --sys-dnbn configs/sys_dnbn_custom.json
```

## Key Configuration Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| M | dnbn config | 512 | Number of task neurons |
| C | dnbn config | 128 | Number of communication neurons |
| rate | connection | 1.0 | Communication frequency (1.0 = every step) |
| send_bandwidth | connection | C | Number of comm neurons allocated to this link |
| send_offset | connection | 0 | Start index in sender's comm vector |
| recv_offset | connection | 0 | Start index in receiver's comm buffer |
| communication_rounds | training | 3 | Message exchange iterations per forward pass |
| bond_sparsity_lambda | training | 0.01 | L1 penalty strength on bond weights |

## Cleanup

```bash
./Neptun.sh --clean
```

Removes the virtual environment and downloaded datasets.
