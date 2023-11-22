from __future__ import annotations

from math import log2
from typing import Tuple

import torch

from pyqtorch.abstract import AbstractOperator
from pyqtorch.apply import apply_operator
from pyqtorch.matrices import OPERATIONS_DICT, _controlled, _dagger
from pyqtorch.utils import Operator, State


class Primitive(AbstractOperator):
    def __init__(self, pauli: torch.Tensor, target: int):
        super().__init__(target)
        self.register_buffer("pauli", pauli)
        self._param_type = None

    @property
    def param_type(self) -> None:
        return self._param_type

    def unitary(self, values: dict[str, torch.Tensor] | torch.Tensor = {}) -> Operator:
        return self.pauli.unsqueeze(2)

    def forward(self, state: State, values: dict[str, torch.Tensor] | torch.Tensor = {}) -> State:
        return apply_operator(
            state, self.unitary(values), self.qubit_support, len(state.size()) - 1
        )

    def dagger(self, values: dict[str, torch.Tensor] | torch.Tensor = {}) -> Operator:
        return _dagger(self.unitary(values))


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

    def forward(self, state: State, values: dict[str, torch.Tensor] = None) -> State:
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


class Projector(Primitive):
    def __init__(self, target: int, state: str = "1"):
        if state == "0":
            super().__init__(OPERATIONS_DICT["PROJ0"], target)
        else:
            super().__init__(OPERATIONS_DICT["PROJ1"], target)


class N(Primitive):
    def __init__(self, target: int):
        super().__init__(OPERATIONS_DICT["N"], target)


class SWAP(Primitive):
    def __init__(self, control: int, target: int):
        super().__init__(OPERATIONS_DICT["SWAP"], target)
        self.control = (control,) if isinstance(control, int) else control
        self.qubit_support = self.control + (target,)
        self.n_qubits = max(self.qubit_support)


class CSWAP(Primitive):
    def __init__(self, control: int | Tuple[int, ...], target: int):
        super().__init__(OPERATIONS_DICT["CSWAP"], target)
        self.control = (control,) if isinstance(control, int) else control
        self.target = target
        self.qubit_support = self.control + (target,)
        self.n_qubits = max(self.qubit_support)


class ControlledOperationGate(Primitive):
    def __init__(self, gate: str, control: int | Tuple[int, ...], target: int):
        self.control = (control,) if isinstance(control, int) else control
        mat = OPERATIONS_DICT[gate]
        mat = _controlled(
            unitary=mat.unsqueeze(2),
            batch_size=1,
            n_control_qubits=len(self.control) - (int)(log2(mat.shape[0])) + 1,
        ).squeeze(2)
        super().__init__(mat, target)
        self.qubit_support = self.control + (target,)
        self.n_qubits = max(self.qubit_support)


class CNOT(ControlledOperationGate):
    def __init__(self, control: int | Tuple[int, ...], target: int):
        super().__init__("X", control, target)


CX = CNOT


class CY(ControlledOperationGate):
    def __init__(self, control: int | Tuple[int, ...], target: int):
        super().__init__("Y", control, target)


class CZ(ControlledOperationGate):
    def __init__(self, control: int | Tuple[int, ...], target: int):
        super().__init__("Z", control, target)


class Toffoli(ControlledOperationGate):
    def __init__(self, control: int | Tuple[int, ...], target: int):
        super().__init__("X", control, target)
