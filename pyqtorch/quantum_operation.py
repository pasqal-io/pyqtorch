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
from pyqtorch.noise import DigitalNoiseProtocol, _repr_noise
from pyqtorch.qubit_support import Support
from pyqtorch.utils import (
    DensityMatrix,
    density_mat,
    expand_operator,
    is_diag_batched,
    permute_basis,
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
        noise ( NoiseProtocol | dict[str, NoiseProtocol. optional): Type of noise
            to add in the operation.
        diagonal (bool, optional): Specify if the operation is diagonal.
            For supporting, only pass a 2-dim operation tensor
            containing diagonal elements with batchsize.

    """

    def __init__(
        self,
        operation: Tensor,
        qubit_support: int | tuple[int, ...] | Support,
        operator_function: Callable | None = None,
        noise: DigitalNoiseProtocol | None = None,
        diagonal: bool = False,
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
        # to inform on diagonality of operation
        self._diagonal = diagonal
        self.is_diagonal: bool = False
        if diagonal:
            if len(operation.size()) != 2:
                raise ValueError(
                    "Only pass diagonal operators as 2D tensors, with batchsize."
                )
            self.is_diagonal = diagonal
        else:
            self.is_diagonal = is_diag_batched(operation)

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
        diagonal: bool = False,
    ) -> Tensor:
        """Get eigenvalues of the tensor of QuantumOperation.

        Args:
            values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            embedding (Embedding | None, optional): Optional embedding. Defaults to None.

        Returns:
            Eigenvalues of the related tensor.
        """
        blockmat = self.tensor(values or dict(), embedding)
        if len(blockmat.shape) == 3:
            return torch.linalg.eigvals(blockmat.permute((2, 0, 1))).reshape(-1, 1)
        else:
            # for diagonal cases
            return blockmat

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
            if (len(self.operation.shape) == 2 and not self._diagonal)
            else self.operation
        )
        return operation

    def dagger(
        self,
        values: dict[str, Tensor] | Tensor | None = None,
        embedding: Embedding | None = None,
        diagonal: bool = False,
    ) -> Tensor:
        """Apply the dagger to operator.

        Args:
            values (dict[str, Tensor] | Tensor, optional): Parameter values. Defaults to dict().
            embedding (Embedding | None, optional): Optional embedding. Defaults to None.
            diagonal (bool, optional): Whether to return the diagonal form of the tensor or not.
                Defaults to False.

        Returns:
            Tensor: conjugate transpose operator.
        """
        blockmat = self.operator_function(values or dict(), embedding)
        use_diagonal = diagonal and self.is_diagonal
        return _dagger(blockmat, use_diagonal)

    def tensor(
        self,
        values: dict[str, Tensor] | None = None,
        embedding: Embedding | None = None,
        full_support: tuple[int, ...] | None = None,
        diagonal: bool = False,
    ) -> Tensor:
        """Get tensor of the QuantumOperation.

        Args:
            values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            embedding (Embedding | None, optional): Optional embedding. Defaults to None.
            full_support (tuple[int, ...] | None, optional): The qubits the returned tensor
                will be defined over. Defaults to None for only using the qubit_support.
            diagonal (bool, optional): Whether to return the diagonal form of the tensor or not.
                Defaults to False.

        Returns:
            Tensor: Tensor representation of QuantumOperation.
        """
        values = values or dict()
        blockmat = self.operator_function(values, embedding)
        use_diagonal = diagonal and self.is_diagonal
        if use_diagonal and not self._diagonal:
            blockmat = torch.diagonal(blockmat).T
        if self._qubit_support.qubits != self.qubit_support:
            blockmat = permute_basis(
                blockmat, self._qubit_support.qubits, inv=True, diagonal=use_diagonal
            )
        if full_support is None:
            return blockmat
        else:
            return expand_operator(
                blockmat, self.qubit_support, full_support, diagonal=use_diagonal
            )

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
