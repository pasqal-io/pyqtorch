from __future__ import annotations

from pyqtorch.matrices import OPERATIONS_DICT
from pyqtorch.noise import DigitalNoiseProtocol
from pyqtorch.quantum_operation import Support
from pyqtorch.utils import (
    product_state,
    qubit_support_as_tuple,
)

from .primitive import ControlledPrimitive, Primitive


class X(Primitive):
    def __init__(
        self,
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(OPERATIONS_DICT["X"], target, noise=noise)


class Y(Primitive):
    def __init__(
        self,
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(OPERATIONS_DICT["Y"], target, noise=noise)


class Z(Primitive):
    def __init__(
        self,
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(OPERATIONS_DICT["Z"], target, noise=noise)


class I(Primitive):  # noqa: E742
    def __init__(
        self,
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(OPERATIONS_DICT["I"], target, noise=noise)


class H(Primitive):
    def __init__(
        self,
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(OPERATIONS_DICT["H"], target, noise=noise)


class T(Primitive):
    def __init__(
        self,
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(OPERATIONS_DICT["T"], target, noise=noise)


class S(Primitive):
    def __init__(
        self,
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(
            OPERATIONS_DICT["S"], target, 0.5 * OPERATIONS_DICT["Z"], noise=noise
        )


class SDagger(Primitive):
    def __init__(
        self,
        target: int,
        noise: DigitalNoiseProtocol | None = None,
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
        noise: DigitalNoiseProtocol | None = None,
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
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(OPERATIONS_DICT["N"], target, noise=noise)


class SWAP(Primitive):
    def __init__(
        self,
        i: int,
        j: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(OPERATIONS_DICT["SWAP"], (i, j), noise=noise)


class CSWAP(Primitive):
    def __init__(
        self,
        control: int,
        target: tuple[int, ...],
        noise: DigitalNoiseProtocol | None = None,
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
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__("X", control, target, noise=noise)


CX = CNOT


class CY(ControlledPrimitive):
    def __init__(
        self,
        control: int | tuple[int, ...],
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__("Y", control, target, noise=noise)


class CZ(ControlledPrimitive):
    def __init__(
        self,
        control: int | tuple[int, ...],
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__("Z", control, target, noise=noise)


class Toffoli(ControlledPrimitive):
    def __init__(
        self,
        control: int | tuple[int, ...],
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__("X", control, target, noise=noise)


OPS_PAULI = {X, Y, Z, I}
OPS_1Q = OPS_PAULI.union({H, S, T})
OPS_2Q = {CNOT, CY, CZ, SWAP}
OPS_3Q = {Toffoli, CSWAP}
OPS_DIGITAL = OPS_1Q.union(OPS_2Q, OPS_3Q)

OPS_DIAGONAL = {Z, I, S, T, SDagger, N, CZ}
