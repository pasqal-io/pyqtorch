from __future__ import annotations

from typing import Any

import torch
from numpy.typing import ArrayLike
from torch.nn import Module

from pyqtorch.core.operation import _apply_gate, create_controlled_matrix_from_operation
from pyqtorch.core.utils import OPERATIONS_DICT


class PrimitiveGate(Module):
    def __init__(self, gate: str, qubits: ArrayLike, n_qubits: int):
        super().__init__()
        self.gate = gate
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.register_buffer("matrix", OPERATIONS_DICT[gate])

    def matrices(self, thetas: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.matrix

    def apply(self, matrix: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return _apply_gate(state, matrix, self.qubits, self.n_qubits)

    def forward(self, _: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        return self.apply(self.matrix, state)

    @property
    def device(self) -> torch.device:
        return self.matrix.device

    def extra_repr(self) -> str:
        return f"'{self.gate}', {self.qubits}, {self.n_qubits}"


def X(*args: Any, **kwargs: Any) -> PrimitiveGate:
    return PrimitiveGate("X", *args, **kwargs)


def Y(*args: Any, **kwargs: Any) -> PrimitiveGate:
    return PrimitiveGate("Y", *args, **kwargs)


def Z(*args: Any, **kwargs: Any) -> PrimitiveGate:
    return PrimitiveGate("Z", *args, **kwargs)


# FIXME: do we really have to apply a matrix here?
# can't we just return the identical state?
def I(*args: Any, **kwargs: Any) -> PrimitiveGate:  # noqa: E743
    return PrimitiveGate("I", *args, **kwargs)


def H(*args: Any, **kwargs: Any) -> PrimitiveGate:
    return PrimitiveGate("H", *args, **kwargs)


def T(*args: Any, **kwargs: Any) -> PrimitiveGate:
    return PrimitiveGate("T", *args, **kwargs)


def S(*args: Any, **kwargs: Any) -> PrimitiveGate:
    return PrimitiveGate("S", *args, **kwargs)


def SWAP(*args: Any, **kwargs: Any) -> PrimitiveGate:
    return PrimitiveGate("SWAP", *args, **kwargs)


class ControlledOperationGate(Module):
    def __init__(self, gate: str, qubits: ArrayLike, n_qubits: int):
        super().__init__()
        self.gate = gate
        self.qubits = qubits
        self.n_qubits = n_qubits
        mat = OPERATIONS_DICT[gate]
        self.register_buffer("matrix", create_controlled_matrix_from_operation(mat))

    def forward(self, _: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        return _apply_gate(state, self.matrix, self.qubits, self.n_qubits)

    @property
    def device(self) -> torch.device:
        return self.matrix.device

    def extra_repr(self) -> str:
        return f"'{self.gate}', {self.qubits}, {self.n_qubits}"


def CNOT(qubits: ArrayLike, n_qubits: int) -> ControlledOperationGate:
    return ControlledOperationGate("X", qubits, n_qubits)
