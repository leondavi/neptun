# DNBN Theory

## Motivation

Traditional neural networks learn features in activations through fixed, uniform connectivity. Each layer connects to the next in a dense, predetermined pattern. DNBN challenges this by introducing a **communication domain** — a learnable infrastructure that determines not just what features are computed, but how information flows between cooperating neural networks.

The key insight: when a single network has only partial information about a problem, performance can improve by exchanging messages with other networks. The system learns an internal protocol that shapes how information flows during inference.

## Formal Definitions

### Neuron State

Each neuron $i$ at time step $t$ maintains a state:

$$h_i^t = f(h_i^{t-1}, x_i^t, u_i^{t-1})$$

where:
- $h_i^{t-1}$ is the previous state
- $x_i^t$ is the current input
- $u_i^{t-1}$ is the aggregated communication from the previous step

### Message Passing

Outgoing message from neuron $i$ to neuron $j$:

$$m_{i \to j}^t = A_{ij} \cdot \phi(h_i^t, c_{i,\text{send}}^t)$$

where:
- $A_{ij}$ is the learnable bond strength between nodes $i$ and $j$
- $\phi$ is the sender function
- $c_{i,\text{send}}^t$ is the sender's communication state

In practice, DNBN treats $m_{i \to j}^t$ as a learned **vector embedding** in a shared communication space:

$$e_{i \to j}^t = W_e [h_i^t; c_{i,\text{send}}^t] \in \mathbb{R}^{d_c}$$

where $d_c$ is communication dimensionality. This lets different models exchange dense semantic vectors instead of scalar signals.

### Communication Neuron Message Buffer

Each communication neuron maintains a short-term message buffer:

$$B_j^t = \text{BufferUpdate}(B_j^{t-1}, \{e_{i \to j}^t\}_i)$$

The buffer stores recent incoming embeddings, making communication temporal rather than purely instantaneous. A simple form is a fixed-length FIFO queue; a trainable form uses decay and gating:

$$B_j^t = \gamma B_j^{t-1} + (1-\gamma) \sum_i \alpha_{ij}^t e_{i \to j}^t$$

where $\gamma \in [0,1)$ controls memory persistence and $\alpha_{ij}^t$ are attention weights.

### Message Aggregation

Incoming messages to neuron $j$ are aggregated:

$$\tilde{u}_j^t = \sum_i A_{ij}^t \, m_{i \to j}^t$$

With buffered embeddings, aggregation becomes attention-based cooperation:

$$q_j^t = W_q h_j^t, \quad k_i^t = W_k e_{i \to j}^t, \quad v_i^t = W_v e_{i \to j}^t$$

$$\alpha_{ij}^t = \text{softmax}_i\left(\frac{(q_j^t)^T k_i^t}{\sqrt{d_k}}\right), \quad \tilde{u}_j^t = \sum_i \alpha_{ij}^t v_i^t$$

This gives each receiver a selective read over cooperative messages, analogous to cross-attention over peers.

### Receive Filter

The aggregated message is filtered before integration:

$$u_j^t = \psi(\tilde{u}_j^t, c_{j,\text{recv}}^t)$$

where $c_{j,\text{recv}}^t$ is the receiver's communication state.

### Output

The task output is a function of all final neuron states:

$$y^t = g(\{h_i^t\})$$

## Communication Domain

The communication domain for each neuron is:

$$c_i^t = (c_{i,\text{send}}^t, \, c_{i,\text{recv}}^t)$$

This is trainable either as a dynamic latent state or as the output of a smaller controller network. Gradients flow into it because it influences the loss through the forward computation.

The communication domain learns three things:
1. **What** information should be transmitted (message content)
2. **Who** should receive or ignore it (routing/gating)
3. **How much** bandwidth or strength to assign (bond strength)

It also supports a fourth practical capability:
4. **When** to retain or discard information via the message buffer dynamics

Together, these make DNBN communication comparable to a distributed memory-and-attention system.

## Bond Formation

### Learnable Adjacency Matrix

Bonds are represented by a learnable matrix $A$, where $A_{ij}$ controls communication strength from node $i$ to node $j$.

Three progressive methods:

#### 1. Soft Bonds (Recommended Start)
$$A_{ij} \in [0, 1], \quad A_{ij} = \sigma(w_{ij})$$

Train the raw parameter $w_{ij}$ with gradient descent. The sigmoid constraint keeps bonds in a valid range. This is the most stable approach.

#### 2. Hard Bonds with Gates
Learn a gate $g_{ij}$ and threshold it for on/off decisions. Requires gradient relaxations (e.g., straight-through estimator or Gumbel-softmax) since discrete decisions break standard gradients.

#### 3. Prune and Sprout
Periodically remove edges where $A_{ij} < \epsilon$ and generate candidate new edges based on neuron similarity:

$$A_{ij}^{\text{candidate}} = \sigma(q(h_i)^T k(h_j))$$

Keep new edges only if they improve loss. This is the most biologically inspired approach.

### Selective Bond Formation

For targeted connectivity, bond scores derive from neuron features:

$$A_{ij} = \sigma(q(h_i)^T k(h_j))$$

With top-$k$ restriction per neuron to prevent fully-connected (and unstable) topologies.

## Training Objective

The total loss combines task performance with communication efficiency:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_1 \mathcal{L}_{\text{bandwidth}} + \lambda_2 \mathcal{L}_{\text{sparsity}} + \lambda_3 \mathcal{L}_{\text{stability}}$$

where:
- $\mathcal{L}_{\text{task}}$: Cross-entropy or MSE for the primary objective
- $\mathcal{L}_{\text{bandwidth}}$: Penalty on message size/frequency
- $\mathcal{L}_{\text{sparsity}}$: $L_1$ penalty on bond strengths: $\sum_{i,j} |A_{ij}|$
- $\mathcal{L}_{\text{stability}}$: Regularizer for training stability

The sparsity penalty is critical — it pushes weak bonds toward zero, naturally pruning the communication graph to retain only the useful connections.

## Training Loop

1. Initialize parameters for state update, sender, receiver, and output modules
2. Forward pass over time: update neuron states, produce messages, aggregate, compute outputs
3. Compute task loss (cross-entropy for classification)
4. Add communication regularizers (sparsity, bandwidth)
5. Backpropagate through the full unrolled computation graph (truncated BPTT if needed)
6. Update parameters with Adam

## Multi-Model Cooperative Learning

In the Neptun framework, multiple DNBN instances form a system:

- Each instance processes the same input independently
- Communication rounds allow instances to exchange learned representations
- The system loss averages task losses across all instances plus bond sparsity
- Ensemble prediction (averaged logits) typically outperforms individual instances

The cooperative learning dynamic encourages specialization — different instances may learn complementary features and share them through the communication infrastructure, with the bond structure evolving to support the most useful information flows.

### Cooperative Attention Capability

DNBN cooperation can be viewed as a multi-agent attention process:

- Each model emits vector embeddings that summarize its current local belief.
- Other models attend to those embeddings based on task context.
- Message buffers preserve high-value signals across rounds, so delayed evidence can still influence later decisions.

This allows the system to perform distributed evidence integration: one model can specialize in edge/texture cues, another in global shape cues, and attention-weighted communication fuses them into a stronger joint prediction.

### Communication Flow Diagram

```mermaid
flowchart LR
	A[Local Neuron State h_i^t] --> B[Sender Projection W_e]
	B --> C[Message Embedding e_{i->j}^t]
	C --> D[Communication Neuron Buffer B_j^t]
	D --> E[Cooperative Attention alpha_{ij}^t]
	E --> F[Aggregated Message u~_j^t]
	F --> G[Receive Filter psi]
	G --> H[Updated Receiver State h_j^{t+1}]
```

The key cooperative loop is that updated receiver states produce new embeddings in the next round, so attention and message buffer content co-evolve over time.

## Bandwidth and Overlap

Communication neuron capacity (C) is allocated across connections:

- Each connection uses a slice of the sender's C-dimensional output
- Multiple connections can have overlapping bandwidth allocations
- On the receiver side, multiple incoming messages can target overlapping neuron ranges
- Overlapping messages are summed additively, creating shared information channels

When communication neurons use message buffers, overlapping channels accumulate over time as well as across senders. This creates a persistent cooperative workspace where the same representational slice can encode consensus, disagreement, or uncertainty.

This overlap mechanism allows rich, multiplexed communication where a receiving model can integrate signals from multiple sources into the same representational subspace.
