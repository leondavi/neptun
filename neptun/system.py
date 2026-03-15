"""DNBN System -- Recurrent cooperative system of expert nodes.

Communication rounds are treated as recurrent time steps with:
- Hidden state updates: h_i^t = GRU(h_i^{t-1}, features_i, buffer_readout_i)
- Buffer updates: B_i^t = roll(B_i^{t-1}); B_i^t[-1] = aggregated_msg
- Training uses BPTT over T communication rounds (truncated T if needed)

This makes DNBN a proper recurrent cooperative system instead of T-layer-deep
feedforward message passing.
"""

import torch
import torch.nn as nn

from .model import DNBNExpert
from .communication import GraphTransformerComm


class DNBNSystem(nn.Module):
    """A system of expert nodes connected via graph transformer communication."""

    def __init__(self, sys_config, input_channels, output_dim):
        super().__init__()
        self.config = sys_config
        self.communication_rounds = sys_config["training"].get(
            "communication_rounds", 3
        )

        self.nodes = nn.ModuleDict()
        first_M = first_C = first_buffer_size = first_num_heads = None

        for node_id, node_cfg in sys_config["nodes"].items():
            p = node_cfg["params"]
            M = p.get("M", 256)
            C = p.get("C", 256)
            num_heads = p.get("num_heads", 4)
            buffer_size = p.get("buffer_size", p.get("queue_size", 8))
            controller_hidden = p.get("controller_hidden", 64)
            dropout = p.get("dropout", 0.1)

            if first_M is None:
                first_M, first_C = M, C
                first_buffer_size, first_num_heads = buffer_size, num_heads

            self.nodes[node_id] = DNBNExpert(
                input_channels=input_channels, output_dim=output_dim,
                M=M, C=C, num_heads=num_heads,
                controller_hidden=controller_hidden, dropout=dropout,
            )

        self.M, self.C = first_M, first_C
        self.buffer_size = first_buffer_size
        self.num_heads = first_num_heads

        top_k = sys_config["training"].get("top_k", None)
        self.comm = GraphTransformerComm(
            node_ids=list(sys_config["nodes"].keys()),
            M=first_M, C=first_C, num_heads=first_num_heads,
            buffer_size=first_buffer_size,
            initial_connections=sys_config.get("connections", []),
            top_k=top_k,
        )
        self._last_attn_weights = []

    def forward(self, x, step=0):
        """Forward pass: feature extraction then T recurrent communication rounds.

        Args:
            x: [B, channels, H, W] image input (same for all experts)
            step: global training step

        Returns:
            dict  node_id -> [B, output_dim] logits
        """
        batch_size = x.shape[0]
        device = x.device

        # 1. Extract features from all experts (once per forward pass)
        features = {}
        for nid, expert in self.nodes.items():
            features[nid] = expert.extract_features(x)

        # 2. Initialize recurrent states
        h = {nid: features[nid].clone() for nid in self.nodes}
        ctrl_states = {nid: None for nid in self.nodes}
        buffers = {
            nid: torch.zeros(batch_size, self.buffer_size, self.C, device=device)
            for nid in self.nodes
        }

        # 3. T recurrent communication rounds (BPTT through all rounds)
        self._last_attn_weights = []

        for t in range(self.communication_rounds):
            # Controller: produce send/recv gates and attention biases
            send_gates, recv_gates, ctrl_biases = {}, {}, {}
            for nid, expert in self.nodes.items():
                sg, rg, ab, cs = expert.controller(h[nid], ctrl_states[nid])
                send_gates[nid] = sg
                recv_gates[nid] = rg
                ctrl_biases[nid] = ab
                ctrl_states[nid] = cs

            # Q, K, V from current states
            q_dict, k_dict, v_dict = {}, {}, {}
            for nid, expert in self.nodes.items():
                q, k, v = expert.get_qkv(h[nid])
                q_dict[nid], k_dict[nid], v_dict[nid] = q, k, v

            # Multi-head attention message passing + buffer update
            buffer_readouts, buffers, attn_weights = self.comm(
                q_dict, k_dict, v_dict,
                send_gates, recv_gates, ctrl_biases,
                h, buffers, step=t,
            )
            self._last_attn_weights.append(attn_weights)

            # Recurrent state update
            for nid, expert in self.nodes.items():
                h[nid] = expert.update_state(
                    h[nid], features[nid], buffer_readouts[nid]
                )

        # 4. Classification from final states
        outputs = {}
        for nid, expert in self.nodes.items():
            outputs[nid] = expert.classify(h[nid])
        return outputs

    def system_loss(self, outputs, targets, criterion):
        """Combined loss: L_task + lambda_s * L_bond + lambda_c * L_comm_cost."""
        task_loss = sum(criterion(out, targets) for out in outputs.values())
        task_loss = task_loss / len(outputs)

        sparsity_lambda = self.config["training"].get("bond_sparsity_lambda", 0.01)
        sparsity_loss = self.comm.bond_sparsity_loss()

        comm_cost_lambda = self.config["training"].get("comm_cost_lambda", 0.001)
        comm_cost = self.comm.communication_cost_loss(self._last_attn_weights)

        total_loss = (
            task_loss + sparsity_lambda * sparsity_loss + comm_cost_lambda * comm_cost
        )
        return total_loss, task_loss
