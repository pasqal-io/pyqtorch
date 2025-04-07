from __future__ import annotations

from functools import cached_property
from typing import Any, Callable

import torch
from torch import Tensor

from pyqtorch.embed import Embedding
from pyqtorch.matrices import OPERATIONS_DICT, controlled
from pyqtorch.noise import DigitalNoiseProtocol, _repr_noise
from pyqtorch.quantum_operation import QuantumOperation, Support
from pyqtorch.utils import DensityMatrix

from .mutation_utils import (
    mutate_control_slice,
    mutate_revert_modified,
    mutate_separate_target,
)


class Primitive(QuantumOperation):
    """Primitive operators based on a fixed matrix U.


    Attributes:
        operation (Tensor): Matrix U.
        qubit_support: List of qubits the QuantumOperation acts on.
        generator (Tensor): A tensor G s.t. U = exp(-iG).
        noise ( NoiseProtocol | dict[str, NoiseProtocol. optional): Type of noise
            to add in the operation.
        diagonal (bool, optional): Specify if the operation is diagonal.
    """

    def __init__(
        self,
        operation: Tensor,
        qubit_support: int | tuple[int, ...] | Support,
        generator: Tensor | None = None,
        noise: DigitalNoiseProtocol | None = None,
        diagonal: bool = False,
    ) -> None:
        super().__init__(operation, qubit_support, noise=noise, diagonal=diagonal)
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

    @property
    def is_parametric(self) -> bool:
        return False


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
        noise: DigitalNoiseProtocol | None = None,
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


class MutablePrimitive(Primitive):
    """Primitive with a mutation operation via a callable `modifier`
    acting directly on the input state.

    Reference: https://arxiv.org/pdf/2303.01493

    Attributes:
        modifier (Callable): Function to modify the state, so applying a gate effect.
    """

    def __init__(
        self,
        operation: Tensor,
        target: int | tuple[int, ...],
        generator: Tensor | None = None,
        noise: DigitalNoiseProtocol | None = None,
        modifier: Callable = lambda s: s,
    ):
        super().__init__(operation, target, generator=generator, noise=noise)
        self.modifier: Callable = modifier

    def _mutate_state_vector(self, state: Tensor) -> Tensor:
        state_shape = state.shape
        perm, state = mutate_separate_target(state, self.target)

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


class PhaseMutablePrimitive(MutablePrimitive):
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
        self.modifier = self.apply_phase1

    def apply_phase1(self, state) -> Tensor:
        phase_mask = torch.ones_like(state, dtype=state.dtype, device=state.device)
        phase_mask[1, :] = self.operation[1, 1]
        phase_state = state * phase_mask
        return phase_state


class MutableControlledPrimitive(ControlledPrimitive):
    """Controlled primitive with a mutation operation via a callable `modifier`
    acting directly on the input state.

    Reference: https://arxiv.org/pdf/2303.01493

    Attributes:
        modifier (Callable): Function to modify the state, so applying a gate effect.
    """

    def __init__(
        self,
        operation: str | Tensor,
        control: int | tuple[int, ...],
        target: int | tuple[int, ...],
        noise: DigitalNoiseProtocol | None = None,
        modifier: Callable = lambda s: s,
    ):
        super().__init__(operation, control, target, noise=noise)
        self.modifier: Callable = modifier

    def _mutate_state_vector(self, state: Tensor) -> Tensor:
        result = state.clone()

        # first, create a mask to separate states to be modified
        mask = mutate_control_slice(len(state.shape) - 1, self.control)
        control_state = state[mask]
        # second, modify only the controlled states
        result[mask] = self._mutate_control_state(control_state)

        return result

    def _mutate_control_state(self, control_state: Tensor) -> Tensor:
        control_state_shape = control_state.shape
        perm, control_state = mutate_separate_target(control_state, self.target)

        control_state = self.modifier(control_state)
        control_state = mutate_revert_modified(control_state, control_state_shape, perm)
        return control_state

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
