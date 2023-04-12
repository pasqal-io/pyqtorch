from typing import Any
from numpy.typing import ArrayLike

import torch
from torch.nn import Module

from pyqtorch.core.utils import OPERATIONS_DICT
from pyqtorch.core.operation import create_controlled_matrix_from_operation, _apply_gate


class PauliGate(Module):
    def __init__(self, gate: str, qubits: ArrayLike, n_qubits: int):
        super().__init__()
        self.gate = gate
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.register_buffer("pauli", OPERATIONS_DICT[gate])

    def forward(self, _: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        return _apply_gate(state, self.pauli, self.qubits, self.n_qubits)

    @property
    def device(self) -> torch.device:
        return self.pauli.device

    def extra_repr(self) -> str:
        return f"'{self.gate}', {self.qubits}, {self.n_qubits}"


def X(*args: Any, **kwargs: Any) -> PauliGate:
    return PauliGate("X", *args, **kwargs)


def Y(*args: Any, **kwargs: Any) -> PauliGate:
    return PauliGate("Y", *args, **kwargs)


def Z(*args: Any, **kwargs: Any) -> PauliGate:
    return PauliGate("Z", *args, **kwargs)


class ControlledOperationGate(Module):
    def __init__(self, gate: str, qubits: ArrayLike, n_qubits: int):
        super().__init__()
        self.gate = gate
        self.qubits = qubits
        self.n_qubits = n_qubits
        mat = OPERATIONS_DICT[gate]
        self.register_buffer("mat", create_controlled_matrix_from_operation(mat))

    def forward(self, _: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        return _apply_gate(state, self.mat, self.qubits, self.n_qubits)

    @property
    def device(self) -> torch.device:
        return self.mat.device

    def extra_repr(self) -> str:
        return f"'{self.gate}', {self.qubits}, {self.n_qubits}"


def CNOT(qubits: ArrayLike, n_qubits: int) -> ControlledOperationGate:
    return ControlledOperationGate("X", qubits, n_qubits)
