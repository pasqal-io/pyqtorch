from __future__ import annotations

from typing import Any

import torch
from numpy.typing import ArrayLike
from torch.nn import Module

from pyqtorch.core.operation import _apply_gate, create_controlled_matrix_from_operation
from pyqtorch.core.utils import OPERATIONS_DICT


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
