from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from numpy.typing import ArrayLike
from torch.nn import Module

import pyqtorch.modules


class AbstractGate(ABC, Module):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__()
        self.qubits = qubits
        self.n_qubits = n_qubits

    def __mul__(
        self, other: AbstractGate | pyqtorch.modules.QuantumCircuit
    ) -> pyqtorch.modules.QuantumCircuit:
        if isinstance(other, AbstractGate):
            return pyqtorch.modules.QuantumCircuit(
                max(self.n_qubits, other.n_qubits), [self, other]
            )

        if isinstance(other, pyqtorch.modules.QuantumCircuit):
            return pyqtorch.modules.QuantumCircuit(
                max(self.n_qubits, other.n_qubits), [self, *other.operations]
            )

        else:
            return NotImplemented

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

    def is_same_gate(self, other: AbstractGate) -> bool:
        return (
            self.qubits == other.qubits
            and self.n_qubits == other.n_qubits
            and type(self) == type(other)
        )
