from __future__ import annotations

from functools import cached_property
from typing import Any

import torch
from torch import Tensor

from pyqtorch.matrices import OPERATIONS_DICT, controlled
from pyqtorch.noise import DigitalNoiseProtocol, _repr_noise
from pyqtorch.quantum_operation import QuantumOperation, Support


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
