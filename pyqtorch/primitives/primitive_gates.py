from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from pyqtorch.embed import Embedding
from pyqtorch.matrices import OPERATIONS_DICT
from pyqtorch.noise import DigitalNoiseProtocol
from pyqtorch.quantum_operation import Support
from pyqtorch.utils import (
    DensityMatrix,
    product_state,
    qubit_support_as_tuple,
)

from .primitive import ControlledPrimitive, Primitive


def mutate_separate_target(
    state: Tensor, target_qubit: int
) -> tuple[list[int], Tensor]:
    """Create a tensor separating the target components
    for a single-qubit gate for mutating an input state-vector.

    Args:
        state (Tensor): Input state.
        target_qubit (int): Target index.

    Returns:
        tuple[int list[int], Tensor]: The permutation indices with an intermediate state
            with separated target qubit.
    """
    n_qubits = len(state.shape) - 1
    perm = list(range(n_qubits + 1))
    perm[0], perm[target_qubit] = perm[target_qubit], perm[0]

    # Transpose the state
    state = state.permute(perm)

    # Reshape to separate the target qubit
    state = state.reshape(2, -1)
    return perm, state


def mutate_revert_modified(
    state: Tensor, original_shape: tuple[int], perm: list[int]
) -> Tensor:
    """After mutating a state given a single qubit gate, we revert back the new state
    to correspond to the `original_shape`.

    Args:
        state (Tensor): modified state by operation.
        original_shape (tuple[int]): original shape for reshapping.
        perm (list[int]): Permutation indices.

    Returns:
        Tensor: Mutated state.
    """
    # Reshape back to original structure
    state = state.reshape(original_shape)

    # Transpose back to original order
    inverse_perm = [perm.index(i) for i in range(len(perm))]
    return state.permute(inverse_perm)


class MutatePrimitive(Primitive):
    """Primitive with a mutation operation via a callable `modifier`
    acting directly on the input state.

    Reference: https://arxiv.org/pdf/2303.01493
    """

    def __init__(
        self,
        operation: Tensor,
        target: int,
        generator: Tensor | None = None,
        noise: DigitalNoiseProtocol | None = None,
        modifier: Callable = lambda s: s,
    ):
        super().__init__(operation, target, generator=generator, noise=noise)
        self.modifier: Callable = modifier

    def _mutate_state_vector(self, state: Tensor) -> Tensor:
        state_shape = state.shape
        perm, state = mutate_separate_target(state, self.target[0])

        # Swap the rows to implement X gate
        state = self.modifier(state)

        return mutate_revert_modified(state, state_shape, perm)

    def _forward(
        self,
        state: Tensor,
        values: dict[str, Tensor] | Tensor | None = None,
        embedding: Embedding | None = None,
    ) -> Tensor:
        values = values or dict()
        if isinstance(state, DensityMatrix):
            return super()._forward(state, values, embedding)
        else:
            return self._mutate_state_vector(state)


class Phase1MutatePrimitive(MutatePrimitive):
    """Primitive with a mutation operation where we multiply by a phase given
    by the matrix element for the state 1.
    """

    def __init__(
        self,
        operation: Tensor,
        target: int,
        generator: Tensor | None = None,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(operation, target, generator=generator, noise=noise)
        self.modifier = self.apphy_phase1

    def apphy_phase1(self, state) -> Tensor:
        phase_mask = torch.ones_like(state, dtype=state.dtype, device=state.device)
        phase_mask[1, :] = self.operation[1, 1]
        phase_state = state * phase_mask
        return phase_state


class X(MutatePrimitive):
    def __init__(
        self,
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(
            OPERATIONS_DICT["X"],
            target,
            noise=noise,
            modifier=lambda s: torch.flip(s, dims=[0]),
        )


class Y(MutatePrimitive):
    def __init__(
        self,
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(OPERATIONS_DICT["Y"], target, noise=noise)
        self.modifier = self._apply_phases

    def _apply_phases(self, state: Tensor) -> Tensor:
        y_state = torch.zeros_like(state)

        # Y gate:
        # |0⟩ -> i|1⟩
        # |1⟩ -> -i|0⟩
        y_state[0, :] = self.operation[0, 1] * state[1, :]  # i|1⟩
        y_state[1, :] = self.operation[1, 0] * state[0, :]  # -i|0⟩

        return y_state


class Z(Phase1MutatePrimitive):
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


class T(Phase1MutatePrimitive):
    def __init__(
        self,
        target: int,
        noise: DigitalNoiseProtocol | None = None,
    ):
        super().__init__(OPERATIONS_DICT["T"], target, noise=noise)


class S(Phase1MutatePrimitive):
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


class SDagger(Phase1MutatePrimitive):
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
