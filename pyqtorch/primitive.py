from __future__ import annotations

from functools import cached_property
from typing import Any

import torch
from torch import Tensor

from pyqtorch.matrices import OPERATIONS_DICT, controlled
from pyqtorch.noise import NoiseProtocol, _repr_noise
from pyqtorch.quantum_ops import QuantumOperation, Support
from pyqtorch.utils import (
    product_state,
    qubit_support_as_tuple,
)


class Primitive(QuantumOperation):
    """Primitive operators based on a fixed matrix U.


    Attributes:
        operation (Tensor): Matrix U.
        qubit_support: List of qubits the QuantumOperation acts on.
        generator (Tensor): A tensor G s.t. U = exp(-iG).
    """

    def __init__(
        self,
        operation: Tensor,
        qubit_support: int | tuple[int, ...] | Support,
        generator: Tensor | None = None,
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ) -> None:
        super().__init__(operation, qubit_support, noise=noise)
        self.generator = generator

    def to(self, *args: Any, **kwargs: Any) -> Primitive:
        """Do device or dtype conversions.

        Returns:
            Primitive: Converted instance.
        """
        super().to(*args, **kwargs)
        if self.generator is not None:
            self.generator.to(*args, **kwargs)
        return self

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
    """Primitive applied depending on control qubits.

    Attributes:
        operation (Tensor): Unitary tensor U.
        control (int | tuple[int, ...]): List of qubits acting as controls.
        target (int | tuple[int, ...]): List of qubits operations acts on.
    """

    def __init__(
        self,
        operation: str | Tensor,
        control: int | tuple[int, ...],
        target: int | tuple[int, ...],
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        support = Support(target, control)
        if isinstance(operation, str):
            operation = OPERATIONS_DICT[operation]
        operation = controlled(
            operation=operation.unsqueeze(2),
            batch_size=1,
            n_control_qubits=len(support.control),
        ).squeeze(2)
        super().__init__(operation, support, noise=noise)

    def extra_repr(self) -> str:
        return f"control: {self.control}, target: {self.target}" + _repr_noise(
            self.noise
        )


class X(Primitive):
    def __init__(
        self,
        target: int,
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        super().__init__(OPERATIONS_DICT["X"], target, noise=noise)


class Y(Primitive):
    def __init__(
        self,
        target: int,
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        super().__init__(OPERATIONS_DICT["Y"], target, noise=noise)


class Z(Primitive):
    def __init__(
        self,
        target: int,
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        super().__init__(OPERATIONS_DICT["Z"], target, noise=noise)


class I(Primitive):  # noqa: E742
    def __init__(
        self,
        target: int,
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        super().__init__(OPERATIONS_DICT["I"], target, noise=noise)


class H(Primitive):
    def __init__(
        self,
        target: int,
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        super().__init__(OPERATIONS_DICT["H"], target, noise=noise)


class T(Primitive):
    def __init__(
        self,
        target: int,
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        super().__init__(OPERATIONS_DICT["T"], target, noise=noise)


class S(Primitive):
    def __init__(
        self,
        target: int,
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        super().__init__(
            OPERATIONS_DICT["S"], target, 0.5 * OPERATIONS_DICT["Z"], noise=noise
        )


class SDagger(Primitive):
    def __init__(
        self,
        target: int,
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        super().__init__(
            OPERATIONS_DICT["SDAGGER"], target, -0.5 * OPERATIONS_DICT["Z"], noise=noise
        )


class Projector(Primitive):
    def __init__(
        self,
        qubit_support: int | tuple[int, ...],
        ket: str,
        bra: str,
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):

        qubit_support = qubit_support_as_tuple(qubit_support)
        if len(ket) != len(bra):
            raise ValueError("Input ket and bra bitstrings must be of same length.")
        if len(qubit_support) != len(ket):
            raise ValueError(
                "Qubit support must have the same number of qubits of ket and bra states."
            )
        ket_state = product_state(ket).flatten()
        bra_state = product_state(bra).flatten()
        super().__init__(
            OPERATIONS_DICT["PROJ"](ket_state, bra_state), qubit_support, noise=noise
        )


class N(Primitive):
    def __init__(
        self,
        target: int,
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        super().__init__(OPERATIONS_DICT["N"], target, noise=noise)


class SWAP(Primitive):
    def __init__(
        self,
        i: int,
        j: int,
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        super().__init__(OPERATIONS_DICT["SWAP"], (i, j), noise=noise)


class CSWAP(Primitive):
    def __init__(
        self,
        control: int,
        target: tuple[int, ...],
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        if not isinstance(target, tuple) or len(target) != 2:
            raise ValueError("Target qubits must be a tuple with two qubits.")
        support = Support(target=qubit_support_as_tuple(control) + target)
        super().__init__(OPERATIONS_DICT["CSWAP"], support)


class CNOT(ControlledPrimitive):
    def __init__(
        self,
        control: int | tuple[int, ...],
        target: int,
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        super().__init__("X", control, target, noise=noise)


CX = CNOT


class CY(ControlledPrimitive):
    def __init__(
        self,
        control: int | tuple[int, ...],
        target: int,
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        super().__init__("Y", control, target, noise=noise)


class CZ(ControlledPrimitive):
    def __init__(
        self,
        control: int | tuple[int, ...],
        target: int,
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        super().__init__("Z", control, target, noise=noise)


class Toffoli(ControlledPrimitive):
    def __init__(
        self,
        control: int | tuple[int, ...],
        target: int,
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        super().__init__("X", control, target, noise=noise)


OPS_PAULI = {X, Y, Z, I}
OPS_1Q = OPS_PAULI.union({H, S, T})
OPS_2Q = {CNOT, CY, CZ, SWAP}
OPS_3Q = {Toffoli, CSWAP}
OPS_DIGITAL = OPS_1Q.union(OPS_2Q, OPS_3Q)
