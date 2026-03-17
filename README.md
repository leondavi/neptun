# Neptun — Distributed Neural and Bonds Network (DNBN)

## Neptun

Neptun explores DNBN, a **Distributed Neural and Bonds Network** in which multiple neural experts operate as nodes in a sparse communication graph.
Each node is a competent expert with its own backbone, while bonds define how information is exchanged, coordinated, and refined across the network over repeated communication rounds.
Rather than treating inter-node interaction as a fixed stack of feedforward graph layers, DNBN treats communication as a recurrent process in which node states evolve over time through message passing, memory, and learned topology.

The motivation behind DNBN is practical: many real problems are not naturally solved by a single monolithic model.
In distributed sensing, modular perception, multi-stage reasoning, or expert specialization, different subnetworks may hold different local competencies and should cooperate without requiring full dense connectivity or a centralized fusion bottleneck.
DNBN is designed to study that setting directly: how specialized experts can exchange compact learned messages, update their internal state, and progressively improve a collective representation or decision.

In this view, bonds are not just static edges.
They are learnable communication pathways that determine which experts should influence one another, how strongly they should communicate, and how information should propagate through the system over time.
This makes the architecture closer to a cooperative dynamical system of neural experts than to a conventional deep network with a fixed computation path.

DNBN draws on familiar ideas from recurrent graph propagation, attention-based communication, and expert specialization.
Its research focus is the combination of these ideas into a sparse, dynamic, recurrent expert graph where communication itself is a first-class learned process.
Neptun is therefore a research framework for experimenting with neural cooperation, adaptive routing, and temporally evolving communication between specialized models.

## Research Hypothesis

DNBN tests the hypothesis that a graph of cooperating neural networks with learnable communication can outperform or match a single monolithic model or a plain ensemble under similar parameter and compute budgets, especially in settings where each model only has partial information about the task.
Instead of treating connectivity as fixed, DNBN makes "who talks to whom, with what bandwidth, and when" a first-class, trainable part of the model via bond strengths, message buffers, and multiple communication rounds.
Concretely, each DNBN instance forms its own local belief about the input, then refines that belief by exchanging vector messages with other instances before making a prediction, so the system is learning both *what to compute* and *how to cooperate*.

## Novelty Relative to Prior Work

Conceptually, DNBN lives at the intersection of ensembles, mixture-of-experts (MoE), and graph/message-passing networks, but differs from each in an important way.

- **Standard ensembles** do not communicate internally — each member makes an independent prediction and the outputs are aggregated (e.g., by averaging or voting).
- **Mixture-of-experts architectures** [[1](#ref-1), [2](#ref-2)] typically route inputs to experts once via a gating network, without recurrent inter-expert message exchange.
- **Graph neural networks and message-passing models** [[3](#ref-3)] operate on tokens or entities within a single model (e.g., nodes in a molecular graph or patches in an image), not on entire neural networks as communicating peers.
- **Learned communication in multi-agent systems** [[4](#ref-4), [5](#ref-5)] demonstrated that agents can learn differentiable communication protocols end-to-end (RIAL/DIAL), including in mixed cooperative-competitive settings, but these operate in reinforcement-learning environments rather than as supervised-learning model architectures.
- **Distributed neural network architectures** [[6](#ref-6)] have explored bandwidth-efficient inference across sensor nodes, but typically start from a fixed centralized architecture that is split and compressed, rather than learning the communication topology and traffic patterns jointly.

DNBN's novelty is to introduce a *communication domain between full networks*, with learnable bond strengths, bandwidth allocation (including overlapping channels), and iterative communication rounds, so that the system can discover which models should exchange what information and at what strength as part of end-to-end training.
In this view, the communication topology and traffic patterns become trainable objects, regularized by sparsity and bandwidth terms, rather than being hand-designed or left implicit.

## Benchmarking Strategy and Compute Trade-offs

The current experiments on MNIST and CIFAR-10 are intended as smoke tests and sanity checks, not as final evidence; in these settings, DNBN variants currently trail strong CNN baselines such as ResNet-18 and EfficientNet-B0 under the same short training budget.
For research-grade evaluation, a fair benchmark setup should start from strong SOTA backbones and then add DNBN communication on top, comparing against those backbones trained with standard best practices (modern datasets, appropriate epoch counts, learning-rate schedules, and, where customary, pretrained weights plus fine-tuning).
In practice, this means (1) choosing competitive CNN/Transformer baselines for each domain, (2) giving them robust training and hyperparameter tuning, and then (3) measuring whether adding DNBN-style inter-model communication provides gains over both the plain backbone and a non-communicating ensemble of the same backbone under matched capacity and compute.

From a compute perspective, DNBN's current vision setup runs multiple expert backbones in parallel and performs several communication rounds per image, so the system-level forward pass is more expensive than a single standard detector like YOLO.
At the same time, each individual DNBN expert backbone is deliberately much lighter than a full YOLO backbone, so part of the research question is whether a collection of cheaper cooperating experts with communication can match or surpass the effectiveness of one heavier monolithic model for a comparable overall compute budget.
See [docs/results_yolo.md](docs/results_yolo.md) for the detailed DNBN vs. YOLO comparison on STL-10, including parameter breakdowns, FLOP estimates, and inference pipeline analysis.

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
- **Recurrent State GRU**: Hidden state evolves across communication rounds via [BPTT](docs/dnbn_theory.md#backpropagation-through-time-bptt)
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
5. Gradients flow through the entire communication graph via [BPTT](docs/dnbn_theory.md#backpropagation-through-time-bptt)

The trainer supports **cosine annealing LR scheduling**, **gradient clipping**, and **weight decay**.

## Theory

See [docs/dnbn_theory.md](docs/dnbn_theory.md) and [agent/background.md](agent/background.md) for the theoretical foundations — bond formation, communication protocols, and training objectives.

## References

<a id="ref-1"></a>
**[1]** Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR 2017. — Foundational MoE paper introducing sparsely-gated expert routing. ([overview](https://www.datacamp.com/blog/mixture-of-experts-moe))

<a id="ref-2"></a>
**[2]** Fedus, W., Zoph, B., & Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 2022. — Simplified MoE routing by switching to a single expert per token. ([overview](https://blog.desigeek.com/post/2025/01/intro-to-mixture-of-experts/))

<a id="ref-3"></a>
**[3]** Kipf, T. N. & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017. — Introduced GCNs for node-level message passing on graph-structured data. ([overview](https://sassafras13.github.io/GNN/))

<a id="ref-4"></a>
**[4]** Foerster, J. N., Assael, Y. M., de Freitas, N., & Whiteson, S. (2016). *Learning to Communicate with Deep Multi-Agent Reinforcement Learning.* NeurIPS 2016. — Proposed RIAL and DIAL for end-to-end learned communication protocols between cooperating agents. ([arXiv:1605.06676](https://arxiv.org/abs/1605.06676), [pdf](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/foersternips16.pdf))

<a id="ref-5"></a>
**[5]** Vanneste, A., Van Wijnsberghe, W., Vanneste, S., Mets, K., Mercelis, S., Latré, S., & Hellinckx, P. (2021). *Mixed Cooperative-Competitive Communication Using Multi-Agent Reinforcement Learning.* BNAIC/BeNeLearn 2021. — Extended DIAL to mixed cooperative-competitive settings, studying private vs. overheard communication. ([arXiv:2110.15762](https://arxiv.org/abs/2110.15762))

<a id="ref-6"></a>
**[6]** Strypsteen, T. & Bertrand, A. (2022). *Bandwidth-efficient distributed neural network architectures with application to body sensor networks.* arXiv 2022. — Methodology for transforming centralized NNs into bandwidth-constrained distributed architectures across sensor nodes. ([arXiv:2210.07750](https://arxiv.org/abs/2210.07750))

## Citation

If this repository is useful in your research, please cite it using [CITATION.cff](CITATION.cff).

Author profile:
- David Leon: https://scholar.google.com/citations?user=J3YceFQAAAAJ&hl=en

## License

Research use only.
