from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from numpy.typing import ArrayLike
from torch.nn import Module


class AbstractGate(ABC, Module):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__()
        self.qubits = qubits
        self.n_qubits = n_qubits

    @abstractmethod
    def matrices(self, tensors: torch.Tensor) -> torch.Tensor:
        # NOTE: thetas are assumed to be of shape (1,batch_size) or (batch_size,) because we
        # want to allow e.g. (3,batch_size) in the U gate.
        ...

    @abstractmethod
    def apply(self, matrix: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def forward(self, state: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        ...

    def extra_repr(self) -> str:
        return f"qubits={self.qubits}, n_qubits={self.n_qubits}"


def is_diag(H: torch.Tensor) -> bool:
    """
    Returns True if Hamiltonian H is diagonal.
    """
    return len(torch.abs(torch.triu(H, diagonal=1)).to_sparse().coalesce().values()) == 0


def is_real(H: torch.Tensor) -> bool:
    """
    Returns True if Hamiltonian H is real.
    """
    return len(torch.imag(H).to_sparse().coalesce().values()) == 0


def rot_matrices(
    theta: torch.Tensor, P: torch.Tensor, I: torch.Tensor, batch_size: int  # noqa: E741
) -> torch.Tensor:
    """
    Returns:
        torch.Tensor: a batch of gates after applying theta
    """
    cos_t = torch.cos(theta / 2).unsqueeze(0).unsqueeze(1)
    cos_t = cos_t.repeat((2, 2, 1))
    sin_t = torch.sin(theta / 2).unsqueeze(0).unsqueeze(1)
    sin_t = sin_t.repeat((2, 2, 1))

    batch_imat = I.unsqueeze(2).repeat(1, 1, batch_size)
    batch_operation_mat = P.unsqueeze(2).repeat(1, 1, batch_size)

    return cos_t * batch_imat - 1j * sin_t * batch_operation_mat
