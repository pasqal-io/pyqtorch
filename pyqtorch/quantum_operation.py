from __future__ import annotations

import logging
from functools import cached_property
from logging import getLogger
from math import log2
from typing import Any, Callable

import torch
from torch import Tensor

from pyqtorch.apply import apply_operator, apply_operator_dm
from pyqtorch.embed import Embedding
from pyqtorch.matrices import _dagger
from pyqtorch.noise import NoiseProtocol, _repr_noise
from pyqtorch.utils import (
    DensityMatrix,
    density_mat,
    expand_operator,
    permute_basis,
    qubit_support_as_tuple,
)

logger = getLogger(__name__)


def forward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    torch.cuda.nvtx.range_pop()


def pre_forward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    torch.cuda.nvtx.range_push("QuantumOperation.forward")


def backward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    torch.cuda.nvtx.range_pop()


def pre_backward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    torch.cuda.nvtx.range_push("QuantumOperation.backward")


class Support:
    """
    Generic representation of the qubit support. For single qubit operations,
    a multiple index support indicates apply the operation for each index in the
    support.

    Both target and control lists must be ordered!

    Attributes:
       target = Index or indices where the operation is applied.
       control = Index or indices to which the operation is conditioned to.
    """

    def __init__(
        self,
        target: int | tuple[int, ...],
        control: int | tuple[int, ...] | None = None,
    ) -> None:
        self.target = qubit_support_as_tuple(target)
        self.control = qubit_support_as_tuple(control) if control is not None else ()
        # if self.qubits != tuple(set(self.qubits)):
        #    raise ValueError("One or more qubits are defined both as control and target.")

    @classmethod
    def target_all(cls) -> Support:
        return Support(target=())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Support):
            return NotImplemented

        return self.target == other.target and self.control == other.control

    def __len__(self):
        return len(self.qubits)

    @cached_property
    def qubits(self) -> tuple[int, ...]:
        return self.control + self.target

    @cached_property
    def sorted_qubits(self) -> tuple[int, ...]:
        return tuple(sorted(self.qubits))

    def __repr__(self) -> str:
        if not self.target:
            return f"{self.__class__.__name__}.target_all()"

        subspace = f"target: {self.target}"
        if self.control:
            subspace += f", control: {self.control}"

        return f"{self.__class__.__name__}({subspace})"


class QuantumOperation(torch.nn.Module):
    """Generic QuantumOperation class storing a tensor operation to represent either
        a quantum operator or a tensor generator inferring the QuantumOperation.

    Attributes:
        operation (Tensor): Tensor used to infer the QuantumOperation
            directly or indirectly.
        qubit_support (int | tuple[int, ...]): Tuple of qubits
            the QuantumOperation acts on.
        operator_function (Callable | None, optional): Function to generate the base operator
            from operation. If None, we consider returning operation itself.

    """

    def __init__(
        self,
        operation: Tensor,
        qubit_support: int | tuple[int, ...] | Support,
        operator_function: Callable | None = None,
        noise: NoiseProtocol | None = None,
    ) -> None:
        """Initializes QuantumOperation

        Args:
            operation (Tensor): Tensor used to infer the QuantumOperation.
            qubit_support (int | tuple[int, ...]): List of qubits
                the QuantumOperation acts on.

        Raises:
            ValueError: When operation has incompatible shape
                with  qubit_support
        """
        super().__init__()

        self._qubit_support = (
            qubit_support
            if isinstance(qubit_support, Support)
            else Support(target=qubit_support)
        )

        self.register_buffer("operation", operation)
        self._device = self.operation.device
        self._dtype = self.operation.dtype

        is_primitive = operator_function is None
        dim_nomatch = len(self.qubit_support) != int(log2(operation.shape[0]))
        if is_primitive and dim_nomatch:
            raise ValueError(
                "The operation shape should match the length of the qubit_support."
            )

        if operator_function is None:
            self._operator_function = self._default_operator_function
        else:
            self._operator_function = operator_function

        self.noise = noise

        if logger.isEnabledFor(logging.DEBUG):
            # When Debugging let's add logging and NVTX markers
            # WARNING: incurs performance penalty
            self.register_forward_hook(forward_hook, always_call=True)
            self.register_full_backward_hook(backward_hook)
            self.register_forward_pre_hook(pre_forward_hook)
            self.register_full_backward_pre_hook(pre_backward_hook)

    def to(self, *args: Any, **kwargs: Any) -> QuantumOperation:
        """Do device or dtype conversions.

        Returns:
            QuantumOperation: Converted instance.
        """
        super().to(*args, **kwargs)
        self._device = self.operation.device
        self._dtype = self.operation.dtype
        return self

    @property
    def qubit_support(self) -> tuple[int, ...]:
        """Getter qubit_support.

        Returns:
            Support: Tuple of sorted qubits.
        """
        return self._qubit_support.sorted_qubits

    @property
    def target(self) -> tuple[int, ...]:
        """Get target qubits.

        Returns:
            tuple[int, ...]: The target qubits
        """
        return self._qubit_support.target

    @property
    def control(self) -> tuple[int, ...]:
        """Get control qubits.

        Returns:
            tuple[int, ...]: The control qubits
        """
        return self._qubit_support.control

    @property
    def operator_function(self) -> Callable[..., Any]:
        """Getter operator_function.

        Returns:
            Callable[..., Any]: Function for
                getting base operator.
        """
        return self._operator_function

    def __hash__(self) -> int:
        return hash(self.qubit_support)

    def extra_repr(self) -> str:
        return f"target: {self.qubit_support}" + _repr_noise(self.noise)

    @property
    def device(self) -> torch.device:
        """Returns device.

        Returns:
            torch.device: Device.
        """
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Returns dtype.

        Returns:
            torch.dtype: Dtype.
        """
        return self._dtype

    @cached_property
    def eigenvals_generator(self) -> Tensor:
        """Get eigenvalues of the underlying operation.

        Arguments:
            values: Parameter values.

        Returns:
            Eigenvalues of the operation.
        """
        return torch.linalg.eigvalsh(self.operation).reshape(-1, 1)

    @cached_property
    def eigenvalues(
        self,
        values: dict[str, Tensor] | Tensor | None = None,
        embedding: Embedding | None = None,
    ) -> Tensor:
        """Get eigenvalues of the tensor of QuantumOperation.

        Args:
            values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            embedding (Embedding | None, optional): Optional embedding. Defaults to None.

        Returns:
            Eigenvalues of the related tensor.
        """
        blockmat = self.tensor(values or dict(), embedding)
        return torch.linalg.eigvals(blockmat.permute((2, 0, 1))).reshape(-1, 1)

    @cached_property
    def spectral_gap(self) -> Tensor:
        """Difference between the moduli of the two largest eigenvalues of the generator.

        Returns:
            Tensor: Spectral gap value.
        """
        spectrum = self.eigenvals_generator
        diffs = spectrum - spectrum.T
        spectral_gap = torch.unique(torch.abs(torch.tril(diffs)))
        return spectral_gap[spectral_gap.nonzero()]

    def _default_operator_function(
        self,
        values: dict[str, Tensor] | Tensor | None = None,
        embedding: Embedding | None = None,
    ) -> Tensor:
        """Default operator_function simply returns the operation.

        Args:
            values (dict[str, Tensor] | Tensor, optional): Parameter values. Defaults to dict().
            embedding (Embedding | None, optional): Optional embedding. Defaults to None.

        Returns:
            Tensor: Base operator.
        """
        operation = (
            self.operation.unsqueeze(2)
            if len(self.operation.shape) == 2
            else self.operation
        )
        return operation

    def dagger(
        self,
        values: dict[str, Tensor] | Tensor | None = None,
        embedding: Embedding | None = None,
    ) -> Tensor:
        """Apply the dagger to operator.

        Args:
            values (dict[str, Tensor] | Tensor, optional): Parameter values. Defaults to dict().
            embedding (Embedding | None, optional): Optional embedding. Defaults to None.

        Returns:
            Tensor: conjugate transpose operator.
        """
        return _dagger(self.operator_function(values or dict(), embedding))

    def tensor(
        self,
        values: dict[str, Tensor] | None = None,
        embedding: Embedding | None = None,
        full_support: tuple[int, ...] | None = None,
    ) -> Tensor:
        """Get tensor of the QuantumOperation.

        Args:
            values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            embedding (Embedding | None, optional): Optional embedding. Defaults to None.
            full_support (tuple[int, ...] | None, optional): The qubits the returned tensor
                will be defined over. Defaults to None for only using the qubit_support.

        Returns:
            Tensor: Tensor representation of QuantumOperation.
        """
        values = values or dict()
        blockmat = self.operator_function(values, embedding)
        if self._qubit_support.qubits != self.qubit_support:
            blockmat = permute_basis(blockmat, self._qubit_support.qubits, inv=True)
        if full_support is None:
            return blockmat
        else:
            return expand_operator(blockmat, self.qubit_support, full_support)

    def _forward(
        self,
        state: Tensor,
        values: dict[str, Tensor] | Tensor | None = None,
        embedding: Embedding | None = None,
    ) -> Tensor:
        values = values or dict()
        if isinstance(state, DensityMatrix):
            return apply_operator_dm(
                state, self.tensor(values, embedding), self.qubit_support
            )
        else:
            return apply_operator(
                state,
                self.tensor(values, embedding),
                self.qubit_support,
            )

    def _noise_forward(
        self,
        state: Tensor,
        values: dict[str, Tensor] | Tensor | None = None,
        embedding: Embedding | None = None,
    ) -> Tensor:
        values = values or dict()
        if not isinstance(state, DensityMatrix):
            state = density_mat(state)

        state = apply_operator_dm(
            state, self.tensor(values, embedding), self.qubit_support
        )

        for noise_class, noise_info in self.noise.gates:  # type: ignore [union-attr]
            if noise_info.target is None:
                target = self.target if len(self.target) == 1 else self.target[0]
            else:
                target = noise_info.target
            noise_gate = noise_class(
                target=target, error_probability=noise_info.error_probability
            )
            state = noise_gate(state, values)

        return state

    def forward(
        self,
        state: Tensor,
        values: dict[str, Tensor] | Tensor | None = None,
        embedding: Embedding | None = None,
    ) -> Tensor:
        """Apply the operation on input state or density matrix.

        Args:
            state (Tensor): Input state.
            values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            embedding (Embedding | None, optional): Optional embedding. Defaults to None.

        Returns:
            Tensor: _description_
        """
        values = values or dict()
        if self.noise:
            return self._noise_forward(state, values, embedding)
        else:
            return self._forward(state, values, embedding)
