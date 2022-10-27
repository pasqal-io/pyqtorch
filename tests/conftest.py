import random

import numpy as np
import torch
from torch.nn import Module, ModuleList

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

from pyqtorch import RY, QuantumCircuit, batchedRY, measurement


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
    def __init__(self, network, n_qubits=4) -> None:
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
