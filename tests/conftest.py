from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.nn import Module, ModuleList

from pyqtorch import QuantumCircuit, batchedRY, measurement
from pyqtorch.core.measurement import total_magnetization
from pyqtorch.core.operation import CNOT, RX, RY, RZ, H, X, Y, Z

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(not torch.cuda.is_available())


class TestFM(QuantumCircuit):
    def __init__(self, n_qubits: int = 4):
        super().__init__(n_qubits)
        self.qubits = range(n_qubits)

    def forward(self, state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        for i in self.qubits:
            state = RY(x, state, [i], self.n_qubits)
        return state


class TestBatchedFM(QuantumCircuit):
    def __init__(self, n_qubits: int = 4):
        super().__init__(n_qubits)
        self.qubits = range(n_qubits)

    def forward(self, state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        for i in self.qubits:
            state = batchedRY(x[:, 0], state, [i], self.n_qubits)
        return state


class TestNetwork(Module):
    def __init__(self, network: torch.nn.Module, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.network = ModuleList(network)
        self.state = QuantumCircuit(n_qubits)
        self.operator = measurement.total_magnetization

    def forward(self, nx: torch.Tensor) -> torch.Tensor:
        batch_size = len(nx)
        state = self.state.init_state(batch_size=batch_size, device=nx.device)
        for layer in self.network:
            state = layer(state, nx)
        return self.operator(state, self.n_qubits, batch_size)


class TestCircuit(QuantumCircuit):
    def __init__(self, n_qubits: int):
        super().__init__(n_qubits)
        self.theta = nn.Parameter(torch.empty((self.n_qubits,)))

    def forward(self) -> torch.Tensor:
        # initial state
        state = self.init_state()

        # single qubit non-parametric gates
        for i in range(self.n_qubits):
            state = H(state, [i], self.n_qubits)

        for i in range(self.n_qubits):
            state = X(state, [i], self.n_qubits)

        for i in range(self.n_qubits):
            state = Y(state, [i], self.n_qubits)

        for i in range(self.n_qubits):
            state = Z(state, [i], self.n_qubits)

        # single-qubit rotation parametric gates
        for i, t in enumerate(self.theta):
            state = RZ(t, state, [i], self.n_qubits)

        for i, t in enumerate(self.theta):
            state = RY(t, state, [i], self.n_qubits)

        for i, t in enumerate(self.theta):
            state = RX(t, state, [i], self.n_qubits)

        # two-qubits gates
        state = CNOT(state, [0, 1], self.n_qubits)
        state = CNOT(state, [2, 3], self.n_qubits)

        return total_magnetization(state, self.n_qubits, 1)


@pytest.fixture
def test_circuit() -> QuantumCircuit:
    return TestCircuit(4)
