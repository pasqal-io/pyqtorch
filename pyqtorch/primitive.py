from __future__ import annotations

import logging
from logging import getLogger
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pyqtorch.apply import apply_operator, operator_product
from pyqtorch.matrices import (
    IMAT,
    OPERATIONS_DICT,
    _controlled,
    _dagger,
)
from pyqtorch.utils import DensityMatrix, product_state

logger = getLogger(__name__)


def forward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    torch.cuda.nvtx.range_pop()


def pre_forward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    torch.cuda.nvtx.range_push("Primitive.forward")


def backward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    torch.cuda.nvtx.range_pop()


def pre_backward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    torch.cuda.nvtx.range_push("Primitive.backward")


class Primitive(torch.nn.Module):
    def __init__(self, pauli: Tensor, target: int | tuple[int, ...]) -> None:
        super().__init__()
        self.target: int | tuple[int, ...] = target

        self.qubit_support: tuple[int, ...] = (
            (target,) if isinstance(target, int) else target
        )
        if isinstance(target, np.integer):
            self.qubit_support = (target.item(),)
        self.register_buffer("pauli", pauli)
        self._device = self.pauli.device
        self._dtype = self.pauli.dtype

        if logger.isEnabledFor(logging.DEBUG):
            # When Debugging let's add logging and NVTX markers
            # WARNING: incurs performance penalty
            self.register_forward_hook(forward_hook, always_call=True)
            self.register_full_backward_hook(backward_hook)
            self.register_forward_pre_hook(pre_forward_hook)
            self.register_full_backward_pre_hook(pre_backward_hook)

    def __hash__(self) -> int:
        return hash(self.qubit_support)

    def extra_repr(self) -> str:
        return f"{self.qubit_support}"

    def unitary(self, values: dict[str, Tensor] | Tensor = dict()) -> Tensor:
        return self.pauli.unsqueeze(2) if len(self.pauli.shape) == 2 else self.pauli

    def forward(
        self, state: Tensor, values: dict[str, Tensor] | Tensor = dict()
    ) -> Tensor:
        if isinstance(state, DensityMatrix):
            # TODO: fix error type int | tuple[int, ...] expected "int"
            # Only supports single-qubit gates
            return DensityMatrix(
                operator_product(
                    self.unitary(values),
                    operator_product(state, self.dagger(values), self.target),  # type: ignore [arg-type]
                    self.target,  # type: ignore [arg-type]
                )
            )
        else:
            return apply_operator(
                state, self.unitary(values), self.qubit_support, len(state.size()) - 1
            )

    def dagger(self, values: dict[str, Tensor] | Tensor = dict()) -> Tensor:
        return _dagger(self.unitary(values))

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def to(self, *args: Any, **kwargs: Any) -> Primitive:
        super().to(*args, **kwargs)
        self._device = self.pauli.device
        self._dtype = self.pauli.dtype
        return self

    def tensor(
        self, values: dict[str, Tensor] = {}, n_qubits: int = 1, diagonal: bool = False
    ) -> Tensor:
        if diagonal:
            raise NotImplementedError
        blockmat = self.unitary(values)
        if n_qubits == 1:
            return blockmat
        full_sup = tuple(i for i in range(n_qubits))
        support = tuple(sorted(self.qubit_support))
        mat = (
            IMAT.clone().to(self.device).unsqueeze(2)
            if support[0] != full_sup[0]
            else blockmat
        )
        for i in full_sup[1:]:
            if i == support[0]:
                other = blockmat
                mat = torch.kron(mat.contiguous(), other.contiguous())
            elif i not in support:
                other = IMAT.clone().to(self.device).unsqueeze(2)
                mat = torch.kron(mat.contiguous(), other.contiguous())
        return mat


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

    def forward(self, state: Tensor, values: dict[str, Tensor] = dict()) -> Tensor:
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


class CSWAP(Primitive):
    def __init__(self, control: int | tuple[int, ...], target: tuple[int, ...]):
        if not isinstance(target, tuple) or len(target) != 2:
            raise ValueError("Target qubits must be a tuple with two qubits")
        super().__init__(OPERATIONS_DICT["CSWAP"], target)
        self.control = (control,) if isinstance(control, int) else control
        self.target = target
        self.qubit_support = self.control + self.target

    def extra_repr(self) -> str:
        return f"control:{self.control}, target:{self.target}"


class ControlledOperationGate(Primitive):
    def __init__(self, gate: str, control: int | tuple[int, ...], target: int):
        self.control = (control,) if isinstance(control, int) else control
        mat = OPERATIONS_DICT[gate]
        mat = _controlled(
            unitary=mat.unsqueeze(2),
            batch_size=1,
            n_control_qubits=len(self.control),
        ).squeeze(2)
        super().__init__(mat, target)
        self.qubit_support = self.control + (self.target,)  # type: ignore[operator]

    def extra_repr(self) -> str:
        return f"control:{self.control}, target:{(self.target,)}"


class CNOT(ControlledOperationGate):
    def __init__(self, control: int | tuple[int, ...], target: int):
        super().__init__("X", control, target)


CX = CNOT


class CY(ControlledOperationGate):
    def __init__(self, control: int | tuple[int, ...], target: int):
        super().__init__("Y", control, target)


class CZ(ControlledOperationGate):
    def __init__(self, control: int | tuple[int, ...], target: int):
        super().__init__("Z", control, target)


class Toffoli(ControlledOperationGate):
    def __init__(self, control: int | tuple[int, ...], target: int):
        super().__init__("X", control, target)
