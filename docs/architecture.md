# DNBN System Architecture

## Overview

The Neptun DNBN (Distributed Neural and Bonds Network) system is a **graph transformer over expert nodes** — a sparse, dynamic communication graph where each node is a competent expert with its own ConvNet backbone. The system implements multi-head attention message passing, recurrent state evolution via GRU, temporal message buffers, and learnable topology. Communication rounds are formally treated as **recurrent time steps with BPTT**, making DNBN a proper recurrent cooperative system rather than T-layer-deep feedforward message passing.

### Theoretical Positioning

DNBN occupies a specific niche relative to existing architectures:

| Architecture | Mechanism | DNBN Analogy |
|---|---|---|
| Standard Transformers | T layers of static dense attention over tokens | T recurrent steps with **shared** communication params and **evolving** states |
| Graph Neural Networks | Message passing over fixed node features | Message passing over **expert embeddings** with **learned topology** |
| Mixture of Experts | Router selects K experts per token | All experts process input; attention selects whose messages to integrate |

The key differentiator: DNBN does **iterative reasoning** — shared parameters, recurrent state, evolving communication patterns — closer to deliberation than a single feedforward pass.

## Architecture Layers

### 1. Expert Node (`DNBNExpert`)

Each expert is a self-contained model with four components:

#### ConvNet Backbone

A small ResNet-like feature extractor replacing the old shallow MLP:

```
Input [B, C_in, H, W]
  → Conv(C_in, 32) + BN + ReLU + ResBlock(32) + MaxPool
  → Conv(32, 64) + BN + ReLU + ResBlock(64) + MaxPool
  → Conv(64, 128) + BN + ReLU + AdaptiveAvgPool(1)
  → Linear(128, M) → features [B, M]
```

**M is the dimension of a pooled representation**, not the only hidden layer. Each expert is individually competent on the task; DNBN communication is an **amplifier**, not a compensator for a weak backbone.

#### Communication Controller (GRU-based)

A small GRU network that produces communication state, formalizing `c_send` and `c_recv` from the theory:

```
c_t = Controller(h_i^{t-1})

Controller:
  GRU state: s_t = GRUCell(h_i, s_{t-1})
  send_gate = σ(W_send · s_t)     ∈ [0,1]^C  — parameterizes/gates sender
  recv_gate = σ(W_recv · s_t)     ∈ [0,1]^C  — controls message integration
  attn_bias = W_bias · s_t        ∈ ℝ^H      — modulates attention logits per head
```

The controller enables **meta-learning of communication protocols**: the GRU learns when to send, what to listen to, and how to modulate attention, evolving its strategy across communication rounds.

#### Multi-Head Attention Projections

Per-expert Q/K/V projections for graph transformer communication:

```
q_i = W_Q · h_i    [B, C]
k_i = W_K · h_i    [B, C]
v_i = W_V · h_i    [B, C]
```

These projections compute from expert state $h_i$ and communication state — the adjacency matrix $A_{ij}$ is **induced by q-k similarity** (plus learned biases), not independent parameters.

#### Recurrent State GRU

Hidden state updates across communication rounds:

$$h_i^t = \text{GRU}([features_i; buffer\_readout_i], h_i^{t-1})$$

This makes communication rounds into **proper recurrent time steps** with shared parameters and evolving states, trained via BPTT.

### 2. Graph Transformer Communication (`GraphTransformerComm`)

#### Multi-Head Attention Message Passing

For each receiver $j$, compute attention over all senders $S(j)$:

$$\alpha_{ij}^h = \text{softmax}\left(\frac{q_j^h \cdot (k_i^h)^\top}{\sqrt{d_h}} + b_{ij} + c_i^h\right)$$

Where:
- $q_j^h, k_i^h$ are per-head queries and keys from expert states
- $b_{ij}$ is a **learnable bond bias** (structural prior from config topology)
- $c_i^h$ is the sender's controller attention bias (per-head modulation)
- Self-connections are masked out ($\alpha_{ii} = 0$)

Aggregated message:
$$m_j = W_{out} \cdot \text{concat}_h\left(\sum_i \alpha_{ij}^h \cdot (v_i \odot g_i^{send})\right) \odot g_j^{recv}$$

Where $g_i^{send}$ and $g_j^{recv}$ are the controller's send/recv gates.

The old scalar bond parameters become **bias terms in attention logits** rather than the sole weight, preserving their role as learnable structural priors while allowing feature-conditioned soft adjacency to dominate.

#### Top-K Sparsification

Optional per-receiver per-head top-k selection:
- Keep only the $k$ highest attention scores per receiver
- All others masked to $-\infty$ before softmax
- Enforces **sparse, dynamic communication** at the expert level

#### Temporal Message Buffers

Each receiver maintains a temporal buffer $B_i \in \mathbb{R}^{B \times T_{buf} \times C}$:

**Buffer update** (each round $t$):
```
B_i^t = roll(B_i^{t-1}, shift=-1)
B_i^t[:, -1, :] = aggregated_message
```

**Buffer readout** via attention with learnable decay $\gamma$:
```
decay_weights = [γ^T_buf, γ^{T_buf-1}, ..., γ^1]  (oldest→newest)

q_buf = W_Q^{buf} · h_i          (query from current state)
k_buf = W_K^{buf} · B_i^t        (keys from buffer entries)
v_buf = W_V^{buf} · B_i^t ⊙ decay_weights  (decay-weighted values)

readout = Attention(q_buf, k_buf, v_buf)   [B, C]
```

This feeds the **full temporal context** (not just the latest message) into the state update, enabling the system to reason about message history.

#### Topology Learning

The adjacency is **feature-conditioned**, not manually designed:

1. **Soft adjacency**: Induced by $q(h_i)^\top k(h_j)$ similarity — experts that produce compatible representations attend to each other
2. **Bond biases**: Config-defined connections initialize positive biases, serving as structural priors that can be strengthened or weakened by training
3. **Top-k sparsification**: Per-node, per-head edge selection enforces locality
4. **Communication cost**: L1 regularization on attention weights + bond biases penalizes unnecessary edges

The model can discover expert clustering and specialization without manual topology design.

### 3. System Orchestration (`DNBNSystem`)

#### Forward Pass (per batch)

```
1. Feature Extraction (once):
   for each expert i:
     features_i = backbone(x)

2. Initialize:
   h_i^0 = features_i
   ctrl_state_i^0 = None
   buffer_i = zeros [B, T_buf, C]

3. For t = 1..T communication rounds:
   a. Controller: (send_gate, recv_gate, attn_bias, ctrl_state) = controller(h_i, ctrl_state_i)
   b. Q/K/V: q_i, k_i, v_i = projections(h_i)
   c. Multi-head attention: message_j = Σ α_{ij} * (v_i ⊙ send_gate_i) ⊙ recv_gate_j
   d. Buffer update: roll + append message
   e. Buffer readout: attention over buffer with decay
   f. State update: h_i^t = GRU([features_i; readout_i], h_i^{t-1})

4. Output:
   logits_i = classifier(h_i^T)
```

#### Loss Function

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_s \cdot \mathcal{L}_{sparsity} + \lambda_c \cdot \mathcal{L}_{comm\_cost}$$

Where:
- $\mathcal{L}_{task} = \frac{1}{N} \sum_i \text{CE}(logits_i, targets)$ — averaged across all expert nodes
- $\mathcal{L}_{sparsity} = \sum_{i,j} |b_{ij}|$ — L1 on bond biases, promotes structural sparsity
- $\mathcal{L}_{comm\_cost} = \sum_t \sum_{i,j,h} \alpha_{ij}^{h,t} + \sum_{i,j} |b_{ij}|$ — penalizes active edges and bandwidth

### 4. Ensemble Output

At evaluation, logits from all experts are averaged for an ensemble prediction. With diverse experts and learned communication, the ensemble typically outperforms any single expert.

## Configuration

### DNBN Expert Config (`dnbn_default.json`)

```json
{
  "M": 256,
  "C": 256,
  "num_heads": 4,
  "buffer_size": 8,
  "controller_hidden": 64,
  "dropout": 0.1
}
```

### System Config (`sys_dnbn_*.json`)

```json
{
  "name": "System name",
  "nodes": {
    "dnbn_0": {"config": "configs/dnbn_default.json"},
    "dnbn_1": {"config": "configs/dnbn_default.json"}
  },
  "connections": [
    {"from": "dnbn_0", "to": "dnbn_1", "rate": 1.0, "send_bandwidth": 64}
  ],
  "training": {
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.001,
    "communication_rounds": 3,
    "bond_sparsity_lambda": 0.01,
    "comm_cost_lambda": 0.001
  }
}
```

Connections define **initial bond biases** (structural priors). The actual communication topology is learned via attention. The `top_k` training parameter (optional) enables sparse communication.

## Data Flow Diagram

```
┌─────────────────────────────────────┐     ┌─────────────────────────────────────┐
│          Expert Node 0              │     │          Expert Node 1              │
│                                     │     │                                     │
│  x ──► [ConvNet Backbone] ──► f_0   │     │  x ──► [ConvNet Backbone] ──► f_1   │
│                                     │     │                                     │
│  h_0^0 = f_0                        │     │  h_1^0 = f_1                        │
│                                     │     │                                     │
│  ┌─── For t = 1..T ──────────────┐  │     │  ┌─── For t = 1..T ──────────────┐  │
│  │ Controller(h) → gates, bias   │  │     │  │ Controller(h) → gates, bias   │  │
│  │ Q,K,V = projections(h)        │  │     │  │ Q,K,V = projections(h)        │  │
│  │                               │  │     │  │                               │  │
│  │    ┌─────────────────────┐    │  │     │  │    ┌─────────────────────┐    │  │
│  │    │  Multi-Head Attn    │◄───┼──┼─────┼──┼───►│  Multi-Head Attn    │    │  │
│  │    │  + Bond Biases      │    │  │     │  │    │  + Bond Biases      │    │  │
│  │    │  + Top-K Sparse     │    │  │     │  │    │  + Top-K Sparse     │    │  │
│  │    └────────┬────────────┘    │  │     │  │    └────────┬────────────┘    │  │
│  │             ▼                 │  │     │  │             ▼                 │  │
│  │    Buffer Update + Readout    │  │     │  │    Buffer Update + Readout    │  │
│  │             ▼                 │  │     │  │             ▼                 │  │
│  │    h^t = GRU([f; readout], h) │  │     │  │    h^t = GRU([f; readout], h) │  │
│  └───────────────────────────────┘  │     │  └───────────────────────────────┘  │
│                                     │     │                                     │
│  logits_0 = classifier(h_0^T)      │     │  logits_1 = classifier(h_1^T)      │
└─────────────────────────────────────┘     └─────────────────────────────────────┘
```

## Expressive Power Analysis

| Property | Standard Transformer | DNBN |
|---|---|---|
| Layers | T independent layers, different params | T recurrent steps, shared params |
| Attention | Dense, static keys/queries per layer | Sparse, evolving keys/queries per step |
| State | Stateless between layers | Recurrent GRU state + temporal buffers |
| Topology | Full attention (all tokens) | Learned sparse graph (top-k experts) |
| Reasoning | Single forward pass | Iterative refinement through rounds |

DNBN's recurrent nature with evolving states makes it closer to **iterative reasoning** mechanisms, where T rounds of communication allow experts to refine their collective understanding through multiple rounds of deliberation.
