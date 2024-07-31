from __future__ import annotations

import logging
from functools import cached_property
from logging import getLogger
from math import log2
from typing import Any, Callable

import torch
from torch import Tensor

from pyqtorch.apply import apply_operator, operator_product
from pyqtorch.embed import Embedding
from pyqtorch.matrices import _dagger
from pyqtorch.utils import (
    DensityMatrix,
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


class QuantumOperation(torch.nn.Module):
    """Basic QuantumOperation class storing a tensor operation which can represent either
        a quantum operator or a generator inferring the QuantumOperation.

    Note that the methods below are meant

    Attributes:
        operation (Tensor): Tensor used to infer the QuantumOperation
            directly or indirectly.
        qubit_support (int | tuple[int, ...]): List of qubits
            the QuantumOperation acts on.
        operator_function (Callable | None, optional): Function to generate the base operator
            from operation. If None, we consider returning operation itself.

    """

    def __init__(
        self,
        operation: Tensor,
        qubit_support: int | tuple[int, ...],
        operator_function: Callable | None = None,
    ) -> None:
        """Initializes QuantumOperation

        Args:
            operation (Tensor): Tensor used to infer the QuantumOperation.
            qubit_support (int | tuple[int, ...]): List of qubits
                the QuantumOperation acts on.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        qubit_support = qubit_support_as_tuple(qubit_support)

        self._qubit_support = qubit_support

        self.register_buffer("operation", operation)
        self._device = self.operation.device
        self._dtype = self.operation.dtype

        if (operator_function is None) and len(self.qubit_support) != int(
            log2(operation.shape[0])
        ):
            raise ValueError(
                "The operation shape should match the length of the qubit_support."
            )

        if operator_function is None:
            self._operator_function = self._default_operator_function
        else:
            self._operator_function = operator_function

        if logger.isEnabledFor(logging.DEBUG):
            # When Debugging let's add logging and NVTX markers
            # WARNING: incurs performance penalty
            self.register_forward_hook(forward_hook, always_call=True)
            self.register_full_backward_hook(backward_hook)
            self.register_forward_pre_hook(pre_forward_hook)
            self.register_full_backward_pre_hook(pre_backward_hook)

    def to(self, *args: Any, **kwargs: Any) -> QuantumOperation:
        super().to(*args, **kwargs)
        self._device = self.operation.device
        self._dtype = self.operation.dtype
        return self

    @cached_property
    def qubit_support(self) -> tuple[int, ...]:
        return tuple(sorted(self._qubit_support))

    @property
    def operator_function(self) -> Callable[..., Any]:
        return self._operator_function

    def __hash__(self) -> int:
        return hash(self.qubit_support)

    def extra_repr(self) -> str:
        return f"{self.qubit_support}"

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
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
    def spectral_gap(self) -> Tensor:
        """Difference between the moduli of the two largest eigenvalues of the generator.

        Returns:
            Tensor: Spectral gap value.
        """
        spectrum = self.eigenvals_generator
        spectral_gap = torch.unique(torch.abs(torch.tril(spectrum - spectrum.T)))
        return spectral_gap[spectral_gap.nonzero()]

    def _default_operator_function(
        self,
        values: dict[str, Tensor] | Tensor = dict(),
        embedding: Embedding | None = None,
    ) -> Tensor:
        operation = (
            self.operation.unsqueeze(2)
            if len(self.operation.shape) == 2
            else self.operation
        )
        return operation

    def dagger(
        self,
        values: dict[str, Tensor] | Tensor = dict(),
        embedding: Embedding | None = None,
    ) -> Tensor:
        return _dagger(self.operator_function(values, embedding))

    def tensor(
        self,
        values: dict[str, Tensor] = {},
        embedding: Embedding | None = None,
        full_support: tuple[int, ...] | None = None,
        diagonal: bool = False,
    ) -> Tensor:
        if diagonal:
            raise NotImplementedError
        blockmat = self.operator_function(values, embedding)
        if self._qubit_support != self.qubit_support:
            blockmat = permute_basis(blockmat, self._qubit_support)
        if full_support is None:
            return blockmat
        else:
            return expand_operator(blockmat, self.qubit_support, full_support)

    def forward(
        self,
        state: Tensor,
        values: dict[str, Tensor] | Tensor = dict(),
        embedding: Embedding | None = None,
    ) -> Tensor:
        if isinstance(state, DensityMatrix):
            # TODO: fix error type int | tuple[int, ...] expected "int"
            # Only supports single-qubit gates
            return DensityMatrix(
                operator_product(
                    self.tensor(values, embedding),
                    operator_product(state, self.dagger(values, embedding), self.qubit_support[-1]),  # type: ignore [arg-type]
                    self.qubit_support[-1],  # type: ignore [arg-type]
                )
            )
        else:
            return apply_operator(
                state,
                self.tensor(values, embedding),
                self.qubit_support,
                len(state.size()) - 1,
            )
