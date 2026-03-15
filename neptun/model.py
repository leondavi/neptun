"""DNBN Expert Node -- ConvNet backbone with communication controller and recurrent state.

Each expert is a node in a graph of experts (graph transformer over expert nodes).
Architecture per expert:
  - ConvNet backbone: small ResNet-like feature extractor producing M-dim pooled representation
  - Communication controller: GRU-based network producing send/recv gates and attention biases
  - Multi-head attention Q/K/V projections for graph transformer communication
  - GRU cell for recurrent state updates across communication rounds (BPTT)
  - Classifier head for task output from final state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual convolutional block with two 3x3 convs and skip connection."""

    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ConvNetBackbone(nn.Module):
    """Small ResNet-like ConvNet backbone for feature extraction.

    Works for any input resolution (uses AdaptiveAvgPool).
    MNIST: 1x28x28, CIFAR-10: 3x32x32.
    Output: M-dimensional pooled representation.
    """

    def __init__(self, input_channels, M, dropout=0.1):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: input_channels -> 32
            nn.Conv2d(input_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32, dropout),
            nn.MaxPool2d(2),
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64, dropout),
            nn.MaxPool2d(2),
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, M)

    def forward(self, x):
        """x: [B, C_in, H, W] -> [B, M]"""
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.proj(out)


class CommunicationController(nn.Module):
    """Per-expert controller that produces communication state.

    Uses a GRU to maintain controller state across communication rounds.
    Outputs:
      - send_gate: gates outgoing values (parameterizes sender)
      - recv_gate: gates aggregated incoming messages (receive filter)
      - attn_bias: per-head biases that modulate attention logits
    """

    def __init__(self, M, C, num_heads, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(M, hidden_dim)
        self.send_gate_proj = nn.Linear(hidden_dim, C)
        self.recv_gate_proj = nn.Linear(hidden_dim, C)
        self.attn_bias_proj = nn.Linear(hidden_dim, num_heads)

    def forward(self, h, prev_state):
        """
        Args:
            h: [B, M] current expert state
            prev_state: [B, hidden_dim] or None
        Returns:
            send_gate: [B, C] sigmoid-gated
            recv_gate: [B, C] sigmoid-gated
            attn_bias: [B, num_heads] unbounded bias terms
            new_state: [B, hidden_dim]
        """
        if prev_state is None:
            prev_state = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
        new_state = self.gru(h, prev_state)
        send_gate = torch.sigmoid(self.send_gate_proj(new_state))
        recv_gate = torch.sigmoid(self.recv_gate_proj(new_state))
        attn_bias = self.attn_bias_proj(new_state)
        return send_gate, recv_gate, attn_bias, new_state


class DNBNExpert(nn.Module):
    """Single DNBN expert node in a graph of experts.

    Each expert independently extracts features via a ConvNet backbone,
    then participates in T rounds of recurrent graph-transformer communication
    where its state evolves via GRU updates informed by attended messages.
    """

    def __init__(self, input_channels, output_dim, M=256, C=256,
                 num_heads=4, controller_hidden=64, dropout=0.1):
        super().__init__()
        self.M = M
        self.C = C
        self.num_heads = num_heads

        self.backbone = ConvNetBackbone(input_channels, M, dropout)
        self.controller = CommunicationController(M, C, num_heads, controller_hidden)
        self.state_gru = nn.GRUCell(M + C, M)
        self.q_proj = nn.Linear(M, C)
        self.k_proj = nn.Linear(M, C)
        self.v_proj = nn.Linear(M, C)
        self.classifier = nn.Linear(M, output_dim)
        self.drop = nn.Dropout(dropout)

    def extract_features(self, x):
        """Extract M-dim pooled features from image input."""
        return self.backbone(x)

    def get_qkv(self, h):
        """Compute Q, K, V for multi-head attention from state h."""
        return self.q_proj(h), self.k_proj(h), self.v_proj(h)

    def update_state(self, prev_h, features, buffer_readout):
        """Recurrent state update: h^t = GRU([features; buffer_readout], h^{t-1})."""
        gru_input = torch.cat([features, buffer_readout], dim=-1)
        return self.state_gru(gru_input, prev_h)

    def classify(self, h):
        """Produce task logits from current state."""
        return self.classifier(self.drop(h))
