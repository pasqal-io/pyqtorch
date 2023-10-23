from __future__ import annotations

from math import log2

import torch

from pyqtorch.apply import _apply_gate
from pyqtorch.matrices import OPERATIONS_DICT, _dagger, make_controlled
from pyqtorch.operator import Operator


class Primitive(Operator):
    def __init__(self, pauli: torch.Tensor, target: int):
        super().__init__(target)
        self.register_buffer("pauli", pauli)
        self.qubit_support = [self.target]
        self.n_qubits = len(self.qubit_support)
        self.apply_fn = _apply_gate

    def unitary(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.pauli

    def apply_operator(self, operator: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.apply_fn(state, operator, self.qubit_support, len(state.size()) - 1)

    def apply_unitary(self, state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.apply_operator(self.unitary(values), state)

    def forward(self, state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.apply_unitary(state, values)

    def dagger(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return _dagger(self.unitary(values).unsqueeze(2)).squeeze(2)

    def apply_dagger(self, state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.apply_operator(self.dagger(values), state)


class X(Primitive):
    def __init__(self, target: int):
        super().__init__(OPERATIONS_DICT["X"], target)


class Y(Primitive):
    def __init__(self, target: int):
        super().__init__(OPERATIONS_DICT["Y"], target)


class Z(Primitive):
    def __init__(self, target: int):
        super().__init__(OPERATIONS_DICT["Z"], target)


class I(Primitive):  # noqa: E742
    def __init__(self, target: int):
        super().__init__(OPERATIONS_DICT["I"], target)

    def forward(self, state: torch.Tensor, values: dict[str, torch.Tensor] = None) -> torch.Tensor:
        return state


class H(Primitive):
    def __init__(self, target: int):
        super().__init__(OPERATIONS_DICT["H"], target)


class T(Primitive):
    def __init__(self, target: int):
        super().__init__(OPERATIONS_DICT["T"], target)


class S(Primitive):
    def __init__(self, target: int):
        super().__init__(OPERATIONS_DICT["S"], target)


class SDagger(Primitive):
    def __init__(self, target: int):
        super().__init__(OPERATIONS_DICT["SDAGGER"], target)


class N(Primitive):
    def __init__(self, target: int):
        super().__init__(OPERATIONS_DICT["N"], target)


class SWAP(Primitive):
    def __init__(self, control: int, target: int):
        super().__init__(OPERATIONS_DICT["SWAP"], target)
        self.control = [control]
        self.qubit_support = self.control + [target]
        self.n_qubits = len(self.qubit_support)


class ControlledOperationGate(Primitive):
    def __init__(self, gate: str, control: int | list[int], target: int):
        self.control: list[int] = [control] if isinstance(control, int) else control
        mat = OPERATIONS_DICT[gate]
        mat = make_controlled(
            unitary=mat.unsqueeze(2),
            batch_size=1,
            n_control_qubits=len(self.control) - (int)(log2(mat.shape[0])) + 1,
        ).squeeze(2)
        super().__init__(mat, target)
        self.qubit_support = self.control + [target]


class CNOT(ControlledOperationGate):
    def __init__(self, control: int | list[int], target: int):
        super().__init__("X", control, target)


CX = CNOT


class CY(ControlledOperationGate):
    def __init__(self, control: int | list[int], target: int):
        super().__init__("Y", control, target)


class CZ(ControlledOperationGate):
    def __init__(self, control: int | list[int], target: int):
        super().__init__("Z", control, target)


class CSWAP(ControlledOperationGate):
    def __init__(self, control: list[int], target: int):
        super().__init__("SWAP", control, target)


class Toffoli(ControlledOperationGate):
    def __init__(self, control: int | list[int], target: int):
        super().__init__("X", control, target)
