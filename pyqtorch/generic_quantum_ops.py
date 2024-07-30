from typing import Any
from math import log2

from functools import cached_property


import torch
from logging import getLogger
from torch import Tensor
import numpy as np

from pyqtorch.embed import Embedding
from pyqtorch.utils import DensityMatrix, expand_operator, permute_basis, product_state


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
    
    Attributes:
        operation: Tensor used to infer the QuantumOperation
            directly or indirectly.
        qubit_support: List of qubits the QuantumOperation acts on.
        
    """
    def __init__(
        self,
        operation: Tensor,
        qubit_support: int | tuple[int, ...],
    ) -> None:

        qubit_support = (
            (qubit_support,) if isinstance(qubit_support, int) else qubit_support
        )
        if isinstance(qubit_support, np.integer):
            qubit_support = (qubit_support.item(),)
        
        if len(qubit_support) != int(log2(operation.shape[0])):
            raise ValueError("The operation shape should match the legth of the qubit_support.")
        
        self._qubit_support = qubit_support
        self.qubit_support = tuple(sorted(qubit_support))

        self.operation = operation
        self.register_buffer("operation", operation)
        self._device = self.operation.device
        self._dtype = self.operation.dtype
    
    def to(self, *args: Any, **kwargs: Any) -> QuantumOperation:
        super().to(*args, **kwargs)
        self._device = self.operation.device
        self._dtype = self.operation.dtype
        return self
    
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
    
    def tensor(
        self,
        values: dict[str, Tensor] = {},
        embedding: Embedding | None = None,
        full_support: tuple[int, ...] | None = None,
        diagonal: bool = False,
    ) -> Tensor:
        return self.operation
    
    def forward(self, state: Tensor):
        raise NotImplementedError
        