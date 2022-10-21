import random

random.seed(0)
import numpy as np

np.random.seed(0)
import torch
from torch.autograd import grad

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
from pyqtorch.ansatz import AlternateLayerAnsatz
from pyqtorch.core.batched_operation import batchedRY
from pyqtorch.core.circuit import QuantumCircuit
from pyqtorch.core.measurement import total_magnetization
from pyqtorch.core.operation import CNOT, RY
from torch.nn import Module, ModuleList


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
        self.operator = total_magnetization

    def forward(self, nx: torch.Tensor) -> torch.Tensor:
        batch_size = len(nx)
        state = self.state.init_state(batch_size=batch_size, device=nx.device)
        for layer in self.network:
            state = layer(state, nx)
        return self.operator(state, self.n_qubits, batch_size)


# TODO: these are all the same test, would be better to parameterize a test
def test_batched_network():
    ansatz = AlternateLayerAnsatz(n_qubits=4, n_layers=4)
    network = TestNetwork([TestFM(), ansatz])
    batched_network = TestNetwork([TestBatchedFM(), ansatz])
    # to ensure the parameters are the same
    batch_size = 2
    x = torch.linspace(-0.5, 0.5, batch_size).reshape(batch_size, 1).requires_grad_()
    bx = torch.linspace(-0.5, 0.5, batch_size).reshape(batch_size, 1).requires_grad_()
    y0: torch.Tensor = network(x[0])
    y1: torch.Tensor = network(x[1])
    by: torch.Tensor = batched_network(bx)

    gby = grad(by, bx, torch.ones_like(by), create_graph=True)
    gy0 = grad(y0, x, torch.ones_like(y0), create_graph=True)
    gy1 = grad(y1, x, torch.ones_like(y1), create_graph=True)

    assert torch.allclose(by[0], y0)
    assert torch.allclose(by[1], y1)
    assert torch.allclose(gby[0][0], gy0[0][0])
    assert torch.allclose(gby[0][1], gy1[0][1])


def test_batched_fm():
    network = TestNetwork([TestFM()])
    batched_network = TestNetwork([TestBatchedFM()])

    batch_size = 3
    x = torch.linspace(-0.5, 0.5, batch_size).reshape(batch_size, 1).requires_grad_()
    bx = torch.linspace(-0.5, 0.5, batch_size).reshape(batch_size, 1).requires_grad_()

    y0: torch.Tensor = network(x[0])
    y1: torch.Tensor = network(x[1])
    by: torch.Tensor = batched_network(bx)

    gby = grad(by, bx, torch.ones_like(by), create_graph=True)
    gy0 = grad(y0, x, torch.ones_like(y0), create_graph=True)
    gy1 = grad(y1, x, torch.ones_like(y1), create_graph=True)

    # Assert result values are the same for single layer
    assert torch.allclose(by[0], y0)
    assert torch.allclose(by[1], y1)
    # Assert gradients are the same
    assert torch.allclose(gby[0][0], gy0[0][0])
    assert torch.allclose(gby[0][1], gy1[0][1])


def test_batched_ansatz():
    network = TestNetwork(
        network=[AlternateLayerAnsatz(n_qubits=2, n_layers=1)], n_qubits=2
    )

    batch_size = 2
    x = torch.linspace(-0.5, 0.5, batch_size).reshape(batch_size, 1).requires_grad_()
    bx = torch.linspace(-0.5, 0.5, batch_size).reshape(batch_size, 1).requires_grad_()
    y0: torch.Tensor = network(x[0])
    y1: torch.Tensor = network(x[1])
    by: torch.Tensor = network(bx)

    assert torch.allclose(by[0], y0)
    assert torch.allclose(by[1], y1)

