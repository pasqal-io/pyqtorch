from __future__ import annotations

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

    def matrices(self, _: torch.Tensor) -> torch.Tensor:
        return self.matrix

    def apply(self, matrix: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return _apply_gate(state, matrix, self.qubits, self.n_qubits)

    def forward(self, state: torch.Tensor, _: torch.Tensor = None) -> torch.Tensor:
        return self.apply(self.matrix, state)

    def extra_repr(self) -> str:
        return f"qubits={self.qubits}, n_qubits={self.n_qubits}"


class X(PrimitiveGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__("X", qubits, n_qubits)


class Y(PrimitiveGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__("Y", qubits, n_qubits)


class Z(PrimitiveGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__("Z", qubits, n_qubits)


# FIXME: do we really have to apply a matrix here?
# can't we just return the identical state?
class I(PrimitiveGate):  # noqa: E742
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__("I", qubits, n_qubits)


class H(PrimitiveGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__("H", qubits, n_qubits)


class T(PrimitiveGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__("T", qubits, n_qubits)


class S(PrimitiveGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__("S", qubits, n_qubits)


class SWAP(PrimitiveGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__("SWAP", qubits, n_qubits)


class ControlledOperationGate(Module):
    def __init__(self, gate: str, qubits: ArrayLike, n_qubits: int):
        super().__init__()
        self.gate = gate
        self.qubits = qubits
        self.n_qubits = n_qubits
        mat = OPERATIONS_DICT[gate]
        self.register_buffer("matrix", create_controlled_matrix_from_operation(mat))

    def matrices(self, _: torch.Tensor) -> torch.Tensor:
        return self.matrix

    def apply(self, matrix: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return _apply_gate(state, matrix, self.qubits, self.n_qubits)

    def forward(self, state: torch.Tensor, _: torch.Tensor = None) -> torch.Tensor:
        return self.apply(self.matrix, state)

    def extra_repr(self) -> str:
        return f"qubits={self.qubits}, n_qubits={self.n_qubits}"


class CNOT(ControlledOperationGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__("X", qubits, n_qubits)


CX = CNOT


class CY(ControlledOperationGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__("Y", qubits, n_qubits)


class CZ(ControlledOperationGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__("Z", qubits, n_qubits)
