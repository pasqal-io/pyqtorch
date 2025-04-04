from __future__ import annotations

import torch
from torch import Tensor

from pyqtorch.embed import Embedding
from pyqtorch.matrices import OPERATIONS_DICT
from pyqtorch.noise import DigitalNoiseProtocol
from pyqtorch.quantum_operation import Support
from pyqtorch.utils import (
    product_state,
    qubit_support_as_tuple,
)

from .mutation_utils import mutate_control_slice
from .primitive import (
    ControlledPrimitive,
    MutableControlledPrimitive,
    MutablePrimitive,
    PhaseMutablePrimitive,
    Primitive,
)


class X(MutablePrimitive):
    def __init__(
        self,
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(
            OPERATIONS_DICT["X"],
            target,
            noise=noise,
        )

    def _mutate_state_vector(self, state: Tensor) -> Tensor:
        #  X gate flips the values at the specified dimension
        return torch.flip(state, [self.target[0]])


class Y(MutablePrimitive):
    def __init__(
        self,
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(OPERATIONS_DICT["Y"], target, noise=noise)
        self.modifier = self._apply_phases

    def _apply_phases(self, state: Tensor) -> Tensor:
        state = torch.flip(state, dims=[0])
        state[0, :] *= self.operation[0, 1]
        state[1, :] *= self.operation[1, 0]

        return state


class Z(PhaseMutablePrimitive):
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

    def _forward(
        self,
        state: Tensor,
        values: dict[str, Tensor] | Tensor | None = None,
        embedding: Embedding | None = None,
    ) -> Tensor:
        return state


class H(Primitive):
    def __init__(
        self,
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(OPERATIONS_DICT["H"], target, noise=noise)


class T(PhaseMutablePrimitive):
    def __init__(
        self,
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(OPERATIONS_DICT["T"], target, noise=noise)


class S(PhaseMutablePrimitive):
    def __init__(
        self,
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(
            OPERATIONS_DICT["S"],
            target,
            generator=0.5 * OPERATIONS_DICT["Z"],
            noise=noise,
        )


class SDagger(PhaseMutablePrimitive):
    def __init__(
        self,
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(
            OPERATIONS_DICT["SDAGGER"],
            target,
            generator=-0.5 * OPERATIONS_DICT["Z"],
            noise=noise,
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


class N(MutablePrimitive):
    def __init__(
        self,
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(OPERATIONS_DICT["N"], target, noise=noise)

        self.modifier = self._zero_out

    def _zero_out(self, state: Tensor) -> Tensor:
        cloned_state = state.clone()
        cloned_state[0, :] = self.operation[0][0]
        return cloned_state


class SWAP(MutablePrimitive):
    def __init__(
        self,
        i: int,
        j: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(OPERATIONS_DICT["SWAP"], (i, j), noise=noise)

    def _mutate_state_vector(self, state: Tensor) -> Tensor:
        #  SWAP gate flips the values at the specified dimensions
        return state.transpose(self.target[0], self.target[1])


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


class CNOT(MutableControlledPrimitive):
    def __init__(
        self,
        control: int | tuple[int, ...],
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__("X", control, target, noise=noise)

    def _mutate_control_state(self, control_state: Tensor) -> Tensor:
        target_adjusted = self.target[0] - sum(
            (1 for c in self.control if c < self.target[0])
        )
        return torch.flip(control_state, [target_adjusted])


CX = CNOT


class CY(ControlledPrimitive):
    def __init__(
        self,
        control: int | tuple[int, ...],
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__("Y", control, target, noise=noise)


class CZ(MutableControlledPrimitive):
    def __init__(
        self,
        control: int | tuple[int, ...],
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__("Z", control, target, noise=noise)

    def _mutate_state_vector(self, state):
        result = state.clone()
        all_qubits = self.control + self.target
        phase_selector = mutate_control_slice(len(state.shape) - 1, all_qubits)
        result[phase_selector] = -result[phase_selector]
        return result


class Toffoli(MutableControlledPrimitive):
    def __init__(
        self,
        control: int | tuple[int, ...],
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__("X", control, target, noise=noise)

    def _mutate_control_state(self, control_state: Tensor) -> Tensor:
        target_adjusted = self.target[0] - sum(
            (1 for c in self.control if c < self.target[0])
        )
        return torch.flip(control_state, [target_adjusted])


OPS_PAULI = {X, Y, Z, I}
OPS_1Q = OPS_PAULI.union({H, S, T})
OPS_2Q = {CNOT, CY, CZ, SWAP}
OPS_3Q = {Toffoli, CSWAP}
OPS_DIGITAL = OPS_1Q.union(OPS_2Q, OPS_3Q)

OPS_DIAGONAL = {Z, I, S, T, SDagger, N, CZ}
