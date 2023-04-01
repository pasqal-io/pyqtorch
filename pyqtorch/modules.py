from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module
from numpy.typing import ArrayLike

from pyqtorch.core.utils import OPERATIONS_DICT
from pyqtorch.core.batched_operation import _apply_batch_gate
from pyqtorch.core.operation import create_controlled_matrix_from_operation, _apply_gate


class PauliGate(Module):
    def __init__(self, gate: str, qubits: ArrayLike, n_qubits):
        super().__init__()
        self.gate = gate
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.register_buffer("pauli", OPERATIONS_DICT[gate])

    def forward(self, _: Tensor, state: Tensor):
        return _apply_gate(state, self.pauli, self.qubits, self.n_qubits)

    @property
    def device(self):
        return self.pauli.device


def X(qubits, n_qubits):
    return PauliGate("X", qubits, n_qubits)


class RotationGate(Module):
    def __init__(self, gate: str, qubits: ArrayLike, n_qubits: int):
        super().__init__()
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.register_buffer("imat", OPERATIONS_DICT["I"])
        self.register_buffer("paulimat", OPERATIONS_DICT[gate])

    def forward(self, theta: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = len(theta)
        mats = rot_matrices(theta, self.paulimat, self.imat, batch_size)
        return _apply_batch_gate(state, mats, self.qubits, self.n_qubits, batch_size)

    @property
    def device(self):
        return self.imat.device



def rot_matrices(
    theta: torch.Tensor, P: torch.Tensor, I: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """
    Returns:
        torch.Tensor: a batch of gates after applying theta
    """
    cos_t = torch.cos(theta / 2).unsqueeze(0).unsqueeze(1)
    cos_t = cos_t.repeat((2, 2, 1))
    sin_t = torch.sin(theta / 2).unsqueeze(0).unsqueeze(1)
    sin_t = sin_t.repeat((2, 2, 1))

    batch_imat = I.unsqueeze(2).repeat(1, 1, batch_size)
    batch_operation_mat = P.unsqueeze(2).repeat(1, 1, batch_size)

    return cos_t * batch_imat - 1j * sin_t * batch_operation_mat


def RX(qubits, n_qubits, **kwargs):
    return RotationGate("X", qubits, n_qubits, **kwargs)


def RY(qubits, n_qubits, **kwargs):
    return RotationGate("Y", qubits, n_qubits, **kwargs)


def RZ(qubits, n_qubits, **kwargs):
    return RotationGate("Z", qubits, n_qubits, **kwargs)


class ControlledOperationGate(Module):
    def __init__(self, gate: str, qubits: ArrayLike, n_qubits: int):
        super().__init__()
        self.gate = gate
        self.qubits = qubits
        self.n_qubits = n_qubits
        mat = OPERATIONS_DICT[gate]
        self.register_buffer("mat", create_controlled_matrix_from_operation(mat))

    def forward(self, _: Tensor, state: Tensor):
        return _apply_gate(state, self.mat, self.qubits, self.n_qubits)

    @property
    def device(self):
        return self.mat.device


def CNOT(qubits: ArrayLike, n_qubits: int):
    return ControlledOperationGate("X", qubits, n_qubits)


def zero_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> torch.Tensor:
    state = torch.zeros((2**n_qubits, batch_size), dtype=dtype, device=device)
    state[0] = 1
    state = state.reshape([2] * n_qubits + [batch_size])
    return state


def uniform_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> torch.Tensor:
    state = torch.ones((2**n_qubits, batch_size), dtype=dtype, device=device)
    state = state / torch.sqrt(torch.tensor(2**n_qubits))
    state = state.reshape([2] * n_qubits + [batch_size])
    return state


class QuantumCircuit(Module):
    def __init__(self, n_qubits: int, operations: list):
        super().__init__()
        self.n_qubits = n_qubits
        self.operations = torch.nn.ModuleList(operations)

    def forward(self, theta: Tensor, state: Tensor):
        for op in self.operations:
            state = op(theta, state)
        return state

    @property
    def device(self):
        devices = set(op.device for op in self.operations)
        if len(devices) == 1:
            return devices.pop()
        else:
            raise ValueError("only one device supported.")

    def init_state(self, batch_size):
        return zero_state(self.n_qubits, batch_size, device=self.device)


if __name__ == "__main__":
    from pyqtorch.core.batched_operation import batchedRX
    import time

    # print(OPERATIONS_DICT["I"])
    # g = OPERATIONS_DICT["I"].to(device="mps", dtype=torch.cfloat)
    # print(g + g)

    dtype = torch.cdouble
    device = "cpu"
    batch_size = 1_000
    state = uniform_state(2, batch_size=batch_size, device=device, dtype=dtype)
    theta = torch.rand(batch_size, device=device, dtype=dtype)

    qubits = [0]
    n_qubits = 2
    iters = 1_000

    t1 = time.time()
    for _ in range(iters):
        batchedRX(theta, state, qubits, n_qubits)
    print(f"{time.time()-t1}")

    gate = RX(qubits, n_qubits).to(device=device, dtype=dtype)
    # gate = torch.compile(gate)
    t1 = time.time()
    for _ in range(iters):
        gate(theta, state)
    print(f"{time.time()-t1}")
    print(state)

    ops = [X([0], 2), X([1], 2), RX([1], 2), CNOT([0, 1], 2)]
    ops = [RX([0], 2)]
    circ = QuantumCircuit(2, ops).to(device=device, dtype=dtype)

    circ(theta, state)
    t1 = time.time()
    for _ in range(iters):
        circ(theta, state)
    print(f"{time.time()-t1}")
    print(state)

    print()
