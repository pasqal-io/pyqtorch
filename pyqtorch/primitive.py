from __future__ import annotations

from functools import cached_property
from math import log2

import torch
from torch import Tensor

from pyqtorch.embed import Embedding
from pyqtorch.generic_quantum_ops import QuantumOperation
from pyqtorch.matrices import OPERATIONS_DICT, _controlled
from pyqtorch.utils import (
    product_state,
    qubit_support_as_tuple,
)


class Primitive(QuantumOperation):
    """Primitive are fixed quantum operations with a defined


    Attributes:
        operation (Tensor): Unitary tensor U.
        qubit_support: List of qubits the QuantumOperation acts on.
        generator (Tensor): A tensor G s.t. U = exp(-iG).
    """

    def __init__(
        self,
        operation: Tensor,
        qubit_support: int | tuple[int, ...],
        generator: Tensor | None = None,
    ) -> None:
        super().__init__(operation, qubit_support)
        self.generator = generator
        if len(self.qubit_support) != int(log2(operation.shape[0])):
            raise ValueError(
                "The operation shape should match the legth of the qubit_support."
            )

    @cached_property
    def eigenvals_generator(self) -> Tensor:
        """Get eigenvalues of the underlying generator.

        Note that for a primitive, the generator is unclear
        so we execute pass.

        Arguments:
            values: Parameter values.

        Returns:
            Eigenvalues of the generator operator.
        """
        if self.generator is not None:
            return torch.linalg.eigvalsh(self.generator).reshape(-1, 1)
        pass


class ControlledPrimitive(Primitive):
    def __init__(
        self,
        operation: str | Tensor,
        control: int | tuple[int, ...],
        target: int | tuple[int, ...],
    ):
        self.control = qubit_support_as_tuple(control)
        self.target = qubit_support_as_tuple(target)
        if isinstance(operation, str):
            operation = OPERATIONS_DICT[operation]
        operation = _controlled(
            unitary=operation.unsqueeze(2),
            batch_size=1,
            n_control_qubits=len(self.control),
        ).squeeze(2)
        super().__init__(operation, self.control + self.target)

    def extra_repr(self) -> str:
        return f"control:{self.control}, targets:{(self.target,)}"


class X(Primitive):
    def __init__(self, qubit_support: int):
        super().__init__(OPERATIONS_DICT["X"], qubit_support)


class Y(Primitive):
    def __init__(self, qubit_support: int):
        super().__init__(OPERATIONS_DICT["Y"], qubit_support)


class Z(Primitive):
    def __init__(self, qubit_support: int):
        super().__init__(OPERATIONS_DICT["Z"], qubit_support)


class I(Primitive):  # noqa: E742
    def __init__(self, qubit_support: int):
        super().__init__(OPERATIONS_DICT["I"], qubit_support)

    def forward(
        self,
        state: Tensor,
        values: dict[str, Tensor] = dict(),
        embedding: Embedding | None = None,
    ) -> Tensor:
        return state


class H(Primitive):
    def __init__(self, qubit_support: int):
        super().__init__(OPERATIONS_DICT["H"], qubit_support)


class T(Primitive):
    def __init__(self, qubit_support: int):
        super().__init__(OPERATIONS_DICT["T"], qubit_support)


class S(Primitive):
    def __init__(self, qubit_support: int):
        super().__init__(
            OPERATIONS_DICT["S"], qubit_support, 0.5 * OPERATIONS_DICT["Z"]
        )


class SDagger(Primitive):
    def __init__(self, qubit_support: int):
        super().__init__(
            OPERATIONS_DICT["SDAGGER"], qubit_support, -0.5 * OPERATIONS_DICT["Z"]
        )


class Projector(Primitive):
    def __init__(self, qubit_support: int | tuple[int, ...], ket: str, bra: str):

        qubit_support = qubit_support_as_tuple(qubit_support)
        if len(ket) != len(bra):
            raise ValueError("Input ket and bra bitstrings must be of same length.")
        if len(qubit_support) != len(ket):
            raise ValueError(
                "Qubit support must have the same number of qubits of ket and bra states."
            )
        ket_state = product_state(ket).flatten()
        bra_state = product_state(bra).flatten()
        super().__init__(OPERATIONS_DICT["PROJ"](ket_state, bra_state), qubit_support)


class N(Primitive):
    def __init__(self, qubit_support: int):
        super().__init__(OPERATIONS_DICT["N"], qubit_support)


class SWAP(Primitive):
    def __init__(self, i: int, j: int):
        super().__init__(OPERATIONS_DICT["SWAP"], (i, j))


class CSWAP(Primitive):
    def __init__(self, control: int, target: tuple[int, ...]):
        if not isinstance(target, tuple) or len(target) != 2:
            raise ValueError("Target qubits must be a tuple with two qubits")
        self.control = qubit_support_as_tuple(control)
        self.target = target
        super().__init__(OPERATIONS_DICT["CSWAP"], self.control + self.target)


class CNOT(ControlledPrimitive):
    def __init__(self, control: int | tuple[int, ...], target: int):
        super().__init__("X", control, target)


CX = CNOT


class CY(ControlledPrimitive):
    def __init__(self, control: int | tuple[int, ...], target: int):
        super().__init__("Y", control, target)


class CZ(ControlledPrimitive):
    def __init__(self, control: int | tuple[int, ...], target: int):
        super().__init__("Z", control, target)


class Toffoli(ControlledPrimitive):
    def __init__(self, control: int | tuple[int, ...], target: int):
        super().__init__("X", control, target)


OPS_PAULI = {X, Y, Z, I}
OPS_1Q = OPS_PAULI.union({H, S, T})
OPS_2Q = {CNOT, CY, CZ, SWAP}
OPS_3Q = {Toffoli, CSWAP}
OPS_DIGITAL = OPS_1Q.union(OPS_2Q, OPS_3Q)
