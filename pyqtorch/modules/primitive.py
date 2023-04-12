import torch
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

    def forward(self, _: dict[str, torch.Tensor], state: torch.Tensor):
        return _apply_gate(state, self.pauli, self.qubits, self.n_qubits)

    @property
    def device(self):
        return self.pauli.device


def X(*args, **kwargs):
    return PauliGate("X", *args, **kwargs)


def Y(*args, **kwargs):
    return PauliGate("Y", *args, **kwargs)


def Z(*args, **kwargs):
    return PauliGate("Z", *args, **kwargs)


class ControlledOperationGate(Module):
    def __init__(self, gate: str, qubits: ArrayLike, n_qubits: int):
        super().__init__()
        self.gate = gate
        self.qubits = qubits
        self.n_qubits = n_qubits
        mat = OPERATIONS_DICT[gate]
        self.register_buffer("mat", create_controlled_matrix_from_operation(mat))

    def forward(self, _: dict[str, torch.Tensor], state: torch.Tensor):
        return _apply_gate(state, self.mat, self.qubits, self.n_qubits)

    @property
    def device(self):
        return self.mat.device


def CNOT(qubits: ArrayLike, n_qubits: int):
    return ControlledOperationGate("X", qubits, n_qubits)

