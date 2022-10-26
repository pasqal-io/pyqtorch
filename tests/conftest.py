import pytest
import torch
import torch.nn as nn

from pyqtorch.core.circuit import QuantumCircuit
from pyqtorch.core.operation import RX, RY, RZ, CNOT, H, X, Y, Z
from pyqtorch.core.measurement import total_magnetization

torch.manual_seed(42)


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
