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

General DNBN message-passing form:

$$m_{i \to j}^t = A_{ij} \cdot \phi(h_i^t, c_{i,\text{send}}^t)$$

where:
- $A_{ij}$ is the learnable bond strength between nodes $i$ and $j$
- $\phi$ is the sender function
- $c_{i,\text{send}}^t$ is the sender's communication state

In current Neptun implementation, communication is instantiated through graph-transformer projections and controller gating:

$$q_i^t = W_q h_i^t, \quad k_i^t = W_k h_i^t, \quad v_i^t = W_v h_i^t$$

$$\hat{v}_i^t = v_i^t \odot g_{i,\text{send}}^t$$

$$\ell_{ji}^t = \frac{(q_j^t)^T k_i^t}{\sqrt{d_k}} + b_{ji} + \beta_i^t$$

where $g_{i,\text{send}}^t$ is sender gate, $b_{ji}$ is a learnable bond-bias prior, and $\beta_i^t$ is a controller-produced sender-side attention bias.

Interpretation of the sender-gated value equation:

- $v_i^t$ is node $i$'s raw message/value vector at communication step $t$.
- $g_{i,\text{send}}^t$ is a learned sender gate (typically in $[0,1]$ per channel).
- $\odot$ is element-wise multiplication, so each channel of $v_i^t$ is scaled independently.

This means communication is filtered before routing: channels with high gate values are transmitted strongly, while channels with low gate values are attenuated or suppressed.

A useful conceptual interpretation is still to view communication as learned vector embeddings in a shared space:

$$e_{i \to j}^t = W_e [h_i^t; c_{i,\text{send}}^t] \in \mathbb{R}^{d_c}$$

where $d_c$ is communication dimensionality. This lets different models exchange dense semantic vectors instead of scalar signals.

### Communication Neuron Message Buffer

Each communication neuron maintains a short-term message buffer:

$$B_j^t = \text{BufferUpdate}(B_j^{t-1}, \{e_{i \to j}^t\}_i)$$

The buffer stores recent incoming embeddings, making communication temporal rather than purely instantaneous.

Current Neptun implementation uses a fixed-length FIFO update:

$$B_j^t = \text{RollLeft}(B_j^{t-1}), \quad B_j^t[-1] \leftarrow m_j^t$$

where $m_j^t$ is the newly aggregated message for receiver $j$.

Readout from this FIFO buffer is attention-based with learnable decay weighting over buffer positions:

$$r_j^t = \text{AttnReadout}(B_j^t; h_j^t, \gamma)$$

where $\gamma \in [0,1)$ controls temporal weighting in readout.

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

Bonds are represented in current implementation as a learnable bias matrix $B$, where $B_{ji}$ is added to attention logits for communication from node $i$ to node $j$.

Current status in Neptun:

1. **Implemented**: soft bond-bias priors in attention logits (unbounded real-valued parameters, regularized with L1).
2. **Implemented**: optional top-$k$ sparsification in attention routing.
3. **Not yet implemented**: hard bond thresholding and prune/sprout rewiring.

### Selective Bond Formation

For targeted connectivity, attention scores derive from node features:

$$s_{ji} = \frac{q(h_j)^T k(h_i)}{\sqrt{d_k}} + b_{ji}$$

with optional top-$k$ restriction per receiver to prevent fully-connected (and unstable) topologies.

## Training Objective

General DNBN objective can combine task performance with communication efficiency:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_1 \mathcal{L}_{\text{bandwidth}} + \lambda_2 \mathcal{L}_{\text{sparsity}} + \lambda_3 \mathcal{L}_{\text{stability}}$$

where:
- $\mathcal{L}_{\text{task}}$: Cross-entropy or MSE for the primary objective
- $\mathcal{L}_{\text{bandwidth}}$: Penalty on message size/frequency
- $\mathcal{L}_{\text{sparsity}}$: $L_1$ penalty on bond strengths: $\sum_{i,j} |A_{ij}|$
- $\mathcal{L}_{\text{stability}}$: Regularizer for training stability

Current Neptun implementation uses:

$$\mathcal{L}_{\text{impl}} = \mathcal{L}_{\text{task}} + \lambda_s \lVert B \rVert_1 + \lambda_c \mathcal{L}_{\text{comm-proxy}}$$

where $\mathcal{L}_{\text{comm-proxy}}$ is computed from attention outputs and bond magnitudes. This is a practical proxy and not yet a strict bandwidth-accurate simulator.

## Training Loop

1. Initialize parameters for state update, sender, receiver, and output modules
2. Forward pass over time: update neuron states, produce messages, aggregate, compute outputs
3. Compute task loss (cross-entropy for classification)
4. Add communication regularizers (currently bond sparsity + communication proxy)
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
	A[Local Expert State h_i^t] --> B[Q/K/V Projections]
	B --> C[Sender Gate and Bond-Bias Attention]
	C --> D[Aggregated Message m_j^t]
	D --> E[FIFO Buffer Update B_j^t]
	E --> F[Buffer Attention Readout r_j^t]
	F --> G[Receive Gate]
	G --> H[State GRU Update h_j^{t+1}]
```

The key cooperative loop is that updated receiver states produce new embeddings in the next round, so attention and message buffer content co-evolve over time.

## Bandwidth and Overlap

Conceptually, communication neuron capacity (C) can be allocated across connections:

Current Neptun implementation does **not** enforce explicit per-connection slice routing from config fields like `send_bandwidth`, `send_offset`, and `recv_offset`; communication currently uses dense attention plus gating.

With FIFO buffers and attention readout, temporal accumulation still occurs across rounds, but explicit per-connection channel slicing remains a conceptual extension rather than active routing logic.

This overlap mechanism allows rich, multiplexed communication where a receiving model can integrate signals from multiple sources into the same representational subspace.
