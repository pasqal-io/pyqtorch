from __future__ import annotations

from math import log2
from typing import Tuple

import torch

from pyqtorch.apply import apply_operator
from pyqtorch.matrices import OPERATIONS_DICT, _controlled, _dagger
from pyqtorch.utils import Operator, State, product_state


class Primitive(torch.nn.Module):
    def __init__(self, pauli: torch.Tensor, target: int) -> None:
        super().__init__()
        self.target: int = target
        self.qubit_support: Tuple[int, ...] = (target,)
        self.n_qubits: int = max(self.qubit_support)
        self.register_buffer("pauli", pauli)
        self._param_type = None

    def __key(self) -> tuple:
        return self.qubit_support

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.__key() == other.__key()
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.qubit_support)

    def extra_repr(self) -> str:
        return f"qubit_support={self.qubit_support}"

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
    def __init__(self, qubit_support: int | tuple[int, ...], ket: str, bra: str):
        support = (qubit_support,) if isinstance(qubit_support, int) else qubit_support
        if len(ket) != len(bra):
            raise ValueError("Input ket and bra bitstrings must be of same length.")
        ket_state = product_state(ket).flatten()
        bra_state = product_state(bra).flatten()
        super().__init__(OPERATIONS_DICT["PROJ"](ket_state, bra_state), support[-1])
        # Override the attribute in AbstractOperator.
        self.qubit_support = support


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
