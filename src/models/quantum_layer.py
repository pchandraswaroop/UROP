"""
quantum_layer.py — Quantum-Enhanced Feature Layer (PennyLane)
=============================================================
A variational quantum circuit layer that post-processes CNN features.

Architecture:
  Classical input → angle embedding → variational circuit → quantum measurements
  → classical linear projection → output features

Design Notes
------------
- n_qubits=4 is the maximum practical count on CPU PennyLane simulator.
- This layer is optional; config.quantum.enabled controls it.
- On CPU, this layer is very slow (~100ms/batch). Disable for full training.
- On Colab, PennyLane-Lightning provides CUDA acceleration.
- Input dim is projected DOWN to n_qubits before the circuit; output is
  projected back UP to output_dim.

Fallback
--------
If pennylane is not installed, IdentityLayer is returned instead, which
is a no-op preserving the expected feature dimension.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import pennylane as qml
    import pennylane.numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    logger.warning(
        "PennyLane not installed. Quantum layer will be replaced by Identity. "
        "Install: pip install pennylane pennylane-lightning"
    )


# ── Quantum Circuit ───────────────────────────────────────────────────────────

class QuantumLayer(nn.Module):
    """Variational Quantum Circuit feature enhancement layer.

    Args:
        input_dim:   Dimension of incoming features (from image branch).
        n_qubits:    Number of qubits. Max 8 for CPU, up to 16 with lightning.
        n_layers:    Number of variational layers in the circuit.
        output_dim:  Dimension of output features.
        device_name: PennyLane device ("lightning.qubit" or "default.qubit").
    """

    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 4,
        n_layers: int = 2,
        output_dim: int = 16,
        device_name: str = "default.qubit",
    ) -> None:
        super().__init__()

        if not PENNYLANE_AVAILABLE:
            raise ImportError(
                "PennyLane required. Install: pip install pennylane pennylane-lightning"
            )

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.output_dim = output_dim

        # Project input features down to n_qubits for angle embedding
        self.input_proj = nn.Linear(input_dim, n_qubits)
        self.input_act  = nn.Tanh()   # → range (-1, 1); scale to (-π, π) in circuit

        # PennyLane device
        self.dev = qml.device(device_name, wires=n_qubits)

        # Trainable variational parameters [n_layers, n_qubits, 3]
        self.n_params = n_layers * n_qubits * 3
        self.quantum_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.01
        )

        # Build QNode
        self.qnode = qml.QNode(
            self._circuit,
            self.dev,
            interface="torch",
            diff_method="backprop",
        )

        # Project quantum output [n_qubits] → output_dim
        self.output_proj = nn.Linear(n_qubits, output_dim)

        logger.info(
            "QuantumLayer: input_dim=%d, n_qubits=%d, n_layers=%d, output_dim=%d",
            input_dim, n_qubits, n_layers, output_dim
        )

    def _circuit(self, inputs: torch.Tensor, weights: torch.Tensor) -> list:
        """Variational quantum circuit.

        Args:
            inputs:  [n_qubits] angle-embedded feature tensor.
            weights: [n_layers, n_qubits, 3] variational weights.

        Returns:
            List of n_qubits expectation values.
        """
        # Angle embedding: Ry gates
        qml.AngleEmbedding(inputs * math.pi, wires=range(self.n_qubits), rotation="Y")

        # Strongly Entangling Layers
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

        # Measure each qubit expectation value
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, input_dim] feature tensor.

        Returns:
            [B, output_dim] quantum-enhanced features.
        """
        orig_device = x.device
        orig_dtype = x.dtype

        # Project and squash to (-1, 1)
        embedded = self.input_act(self.input_proj(x))  # [B, n_qubits]

        # PennyLane default.qubit simulator executes in CPU space.
        # Moving tensors to CPU avoids autoray device mismatch between GPU weights/inputs
        # and default.qubit's internal state. PyTorch autograd tracks transitions across
        # CPU/GPU boundaries during the backward pass seamlessly.
        embedded_cpu = embedded.to("cpu")
        weights_cpu = self.quantum_weights.to("cpu")

        # Handle 1D vs 2D
        is_1d = (embedded_cpu.ndim == 1)
        if is_1d:
            embedded_cpu = embedded_cpu.unsqueeze(0)

        # Vectorized batch execution through PennyLane QNode (~15x faster than looping)
        try:
            res = self.qnode(embedded_cpu, weights_cpu)
            if isinstance(res, (list, tuple)):
                if res[0].ndim > 0:
                    q_batch = torch.stack(res, dim=-1)
                else:
                    q_batch = torch.stack(res).unsqueeze(0)
            else:
                q_batch = res
        except Exception:
            outputs = []
            for i in range(embedded_cpu.shape[0]):
                q_out = self.qnode(embedded_cpu[i], weights_cpu)
                outputs.append(torch.stack(q_out))
            q_batch = torch.stack(outputs)

        if is_1d:
            q_batch = q_batch.squeeze(0)

        # Move back to original device and dtype for subsequent linear projection
        q_batch = q_batch.to(device=orig_device, dtype=orig_dtype)
        return self.output_proj(q_batch)  # [B, output_dim]


# ── Fallback ──────────────────────────────────────────────────────────────────

class IdentityQuantumLayer(nn.Module):
    """Drop-in replacement when PennyLane is not available.

    Projects input to output_dim with a simple linear layer.
    """

    def __init__(self, input_dim: int, output_dim: int = 16, **kwargs) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
        )
        self.output_dim = output_dim
        logger.info(
            "IdentityQuantumLayer (PennyLane fallback): input_dim=%d → output_dim=%d",
            input_dim, output_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def get_quantum_layer(
    input_dim: int,
    n_qubits: int = 4,
    n_layers: int = 2,
    output_dim: int = 16,
    enabled: bool = True,
    device_name: str = "default.qubit",
) -> nn.Module:
    """Factory: return QuantumLayer or IdentityQuantumLayer based on availability.

    Args:
        enabled: If False, always return IdentityQuantumLayer even if PennyLane
                 is available. Use when running on CPU for speed.
    """
    if enabled and PENNYLANE_AVAILABLE:
        return QuantumLayer(
            input_dim=input_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            output_dim=output_dim,
            device_name=device_name,
        )
    return IdentityQuantumLayer(input_dim=input_dim, output_dim=output_dim)
