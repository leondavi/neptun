"""Graph transformer communication layer for DNBN expert graph.

Implements multi-head attention over expert embeddings with:
- Feature-conditioned soft adjacency via Q/K similarity (replaces scalar bonds)
- Learnable bond biases as structural priors in attention logits
- Top-k sparsification per receiver per head
- Temporal message buffers with attention-based readout and learnable decay
- Communication cost regularization (active edges + bandwidth)
- Topology learning: adjacency is induced by Q-K similarity, not independent params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphTransformerComm(nn.Module):
    """Multi-head attention-based communication over a graph of experts.

    The adjacency matrix A is induced by q-k similarity plus learned biases,
    instead of being an independent parameter. The old scalar bond strengths
    become bias terms in the attention logits, preserving their role as
    learnable structural priors.
    """

    def __init__(self, node_ids, M, C, num_heads=4, buffer_size=8,
                 initial_connections=None, top_k=None):
        super().__init__()
        self.M = M
        self.C = C
        self.num_heads = num_heads
        self.head_dim = C // num_heads
        self.buffer_size = buffer_size
        self.top_k = top_k
        self.node_ids = list(node_ids)
        self.N = len(self.node_ids)
        self.node_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}

        # Bond bias: N x N learnable bias in attention logits
        # Initialized from config connections as structural priors
        self.bond_bias = nn.Parameter(torch.zeros(self.N, self.N))
        if initial_connections:
            for conn in initial_connections:
                src_idx = self.node_to_idx.get(conn["from"])
                tgt_idx = self.node_to_idx.get(conn["to"])
                if src_idx is not None and tgt_idx is not None:
                    # bond_bias[receiver, sender] = positive prior
                    self.bond_bias.data[tgt_idx, src_idx] = 1.0

        # Mask out self-connections
        self.register_buffer("self_mask", torch.eye(self.N, dtype=torch.bool))

        # Output projection after multi-head concat
        self.out_proj = nn.Linear(C, C)

        # Buffer readout attention (shared across nodes)
        # Query from expert state (M-dim), K/V from buffer entries (C-dim)
        self.buffer_q = nn.Linear(M, C)
        self.buffer_k = nn.Linear(C, C)
        self.buffer_v = nn.Linear(C, C)

        # Learnable buffer decay parameter gamma
        self.buffer_decay_logit = nn.Parameter(torch.tensor(2.0))  # sigmoid -> ~0.88

        self.reset_stats()

    def reset_stats(self):
        """Reset accumulated communication statistics."""
        self._stats = {
            "total_rounds": 0,
            "total_active_edges": 0.0,
            "total_possible_edges": 0,
            "total_msg_magnitude": 0.0,
            "total_msg_variance": 0.0,
            "total_msg_max": 0.0,
        }

    def forward(self, q_dict, k_dict, v_dict,
                send_gates, recv_gates, ctrl_biases,
                h_dict, buffers, step=0):
        """Multi-head attention message passing with buffer update and readout.

        Args:
            q_dict, k_dict, v_dict: {node_id: [B, C]} from expert Q/K/V projections
            send_gates: {node_id: [B, C]} from controller (gates outgoing values)
            recv_gates: {node_id: [B, C]} from controller (gates incoming messages)
            ctrl_biases: {node_id: [B, num_heads]} attention biases from controller
            h_dict: {node_id: [B, M]} current expert states (for buffer readout query)
            buffers: {node_id: [B, buffer_size, C]} temporal message buffers
            step: current round index

        Returns:
            buffer_readouts: {node_id: [B, C]}
            new_buffers: {node_id: [B, buffer_size, C]}
            attn_weights: [B, num_heads, N, N]
        """
        batch_size = next(iter(q_dict.values())).shape[0]
        device = next(iter(q_dict.values())).device

        # Stack into tensors [N, B, C]
        Q = torch.stack([q_dict[nid] for nid in self.node_ids])
        K = torch.stack([k_dict[nid] for nid in self.node_ids])
        V = torch.stack([v_dict[nid] for nid in self.node_ids])
        S = torch.stack([send_gates[nid] for nid in self.node_ids])

        # Gate outgoing values by send gate (controller parameterizes sender)
        V = V * S

        # Reshape for multi-head: [N, B, num_heads, head_dim]
        Q = Q.view(self.N, batch_size, self.num_heads, self.head_dim)
        K = K.view(self.N, batch_size, self.num_heads, self.head_dim)
        V = V.view(self.N, batch_size, self.num_heads, self.head_dim)

        # Permute to [B, num_heads, N, head_dim]
        Q = Q.permute(1, 2, 0, 3)
        K = K.permute(1, 2, 0, 3)
        V = V.permute(1, 2, 0, 3)

        # Attention logits: [B, H, N_recv, N_send]
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Add bond bias as structural prior: [N, N] -> [1, 1, N, N]
        attn_logits = attn_logits + self.bond_bias.unsqueeze(0).unsqueeze(0)

        # Add controller biases (sender-side modulation)
        CB = torch.stack([ctrl_biases[nid] for nid in self.node_ids])  # [N, B, H]
        CB = CB.permute(1, 2, 0).unsqueeze(2)  # [B, H, 1, N_send]
        attn_logits = attn_logits + CB

        # Mask self-connections (experts don't attend to themselves)
        self_mask = self.self_mask.unsqueeze(0).unsqueeze(0)
        attn_logits = attn_logits.masked_fill(self_mask, float("-inf"))

        # Top-k sparsification per receiver per head
        if self.top_k is not None and self.top_k < self.N - 1:
            _, topk_idx = attn_logits.topk(self.top_k, dim=-1)
            mask = torch.ones_like(attn_logits, dtype=torch.bool)
            mask.scatter_(-1, topk_idx, False)
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))

        # Softmax -> attention weights
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, H, N, N]

        # Aggregate messages: [B, H, N_recv, head_dim]
        messages = torch.matmul(attn_weights, V)
        # Reshape: [B, H, N, D] -> [B, N, H*D] = [B, N, C]
        messages = messages.permute(0, 2, 1, 3).contiguous().view(
            batch_size, self.N, self.C
        )
        messages = self.out_proj(messages)

        # Apply receive gates (controller decides integration strength)
        R = torch.stack([recv_gates[nid] for nid in self.node_ids])  # [N, B, C]
        R = R.permute(1, 0, 2)  # [B, N, C]
        messages = messages * R

        # Update buffers and compute readouts
        decay = torch.sigmoid(self.buffer_decay_logit)
        buffer_readouts = {}
        new_buffers = {}

        for idx, nid in enumerate(self.node_ids):
            msg = messages[:, idx, :]  # [B, C]
            buf = buffers[nid]  # [B, buffer_size, C]

            # Roll buffer: drop oldest, append new aggregated message
            new_buf = torch.roll(buf, shifts=-1, dims=1)
            new_buf = new_buf.clone()
            new_buf[:, -1, :] = msg
            new_buffers[nid] = new_buf

            # Buffer readout via attention with temporal decay
            # Decay weights: newest (last) = gamma^1, oldest (first) = gamma^buffer_size
            t_weights = decay ** torch.arange(
                self.buffer_size, 0, -1, device=device, dtype=torch.float32
            )

            h_i = h_dict[nid]  # [B, M]
            bq = self.buffer_q(h_i).unsqueeze(1)  # [B, 1, C]
            bk = self.buffer_k(new_buf)  # [B, buffer_size, C]
            bv = self.buffer_v(new_buf) * t_weights.view(1, -1, 1)  # decay-weighted

            buf_attn = torch.matmul(bq, bk.transpose(-2, -1)) / (self.C ** 0.5)
            buf_attn = F.softmax(buf_attn, dim=-1)  # [B, 1, buffer_size]
            readout = torch.matmul(buf_attn, bv).squeeze(1)  # [B, C]

            buffer_readouts[nid] = readout

        # Collect statistics
        with torch.no_grad():
            active = (attn_weights > 1e-4).float().sum().item()
            total = self.N * (self.N - 1) * self.num_heads * batch_size
            msg_mag = messages.norm(dim=-1).mean().item()
            msg_var = messages.var(dim=-1).mean().item()
            msg_max = messages.abs().max().item()

            self._stats["total_rounds"] += 1
            self._stats["total_active_edges"] += active
            self._stats["total_possible_edges"] += total
            self._stats["total_msg_magnitude"] += msg_mag
            self._stats["total_msg_variance"] += msg_var
            self._stats["total_msg_max"] = max(self._stats["total_msg_max"], msg_max)

        return buffer_readouts, new_buffers, attn_weights

    def communication_cost_loss(self, attn_weights_list):
        """Communication cost: penalizes active edges and per-edge bandwidth.

        Args:
            attn_weights_list: list of [B, H, N, N] attention weights per round
        """
        cost = torch.tensor(0.0, device=self.bond_bias.device)
        for aw in attn_weights_list:
            cost = cost + aw.sum()
        cost = cost + self.bond_bias.abs().sum()
        return cost

    def bond_sparsity_loss(self):
        """L1 penalty on bond biases (structural prior regularization)."""
        return self.bond_bias.abs().sum()

    def get_comm_stats(self):
        """Return communication statistics summary."""
        n = max(self._stats["total_rounds"], 1)
        return {
            "total_messages": self._stats["total_rounds"] * self.N,
            "skipped_by_rate": 0,
            "avg_msg_magnitude": self._stats["total_msg_magnitude"] / n,
            "avg_msg_variance": self._stats["total_msg_variance"] / n,
            "max_msg_value": self._stats["total_msg_max"],
            "avg_active_edges_per_round": self._stats["total_active_edges"] / n,
            "total_possible_edges": self._stats["total_possible_edges"],
            "sparsity_ratio": 1.0 - (
                self._stats["total_active_edges"]
                / max(self._stats["total_possible_edges"], 1)
            ),
            "per_connection": {},
            "per_node_sent": {
                nid: int(self._stats["total_rounds"]) for nid in self.node_ids
            },
            "per_node_recv": {
                nid: int(self._stats["total_rounds"]) for nid in self.node_ids
            },
        }

    def get_bond_summary(self):
        """Return bond bias values as dict (compatible with old format)."""
        summary = {}
        for i, nid_recv in enumerate(self.node_ids):
            for j, nid_send in enumerate(self.node_ids):
                if i != j:
                    key = f"{nid_send}__to__{nid_recv}"
                    summary[key] = self.bond_bias[i, j].item()
        return summary

    def get_topology_info(self):
        """Return soft adjacency matrix info for analysis."""
        with torch.no_grad():
            adj = self.bond_bias.detach()
            return {
                "bond_bias_matrix": adj.cpu().tolist(),
                "node_ids": self.node_ids,
                "mean_bias": adj.abs().mean().item(),
                "max_bias": adj.abs().max().item(),
                "positive_edges": int((adj > 0).sum().item()),
            }
