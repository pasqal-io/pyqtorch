from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from numpy.typing import ArrayLike
from torch.nn import Module

import pyqtorch.modules as pyq


class AbstractGate(ABC, Module):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__()
        self.qubits = qubits
        self.n_qubits = n_qubits

    def __mul__(self, other: AbstractGate | pyq.QuantumCircuit) -> pyq.QuantumCircuit:
        if isinstance(other, AbstractGate):
            ml = torch.nn.ModuleList([self, other])
            return pyq.QuantumCircuit(max(self.n_qubits, other.n_qubits), ml)
        elif isinstance(other, pyq.QuantumCircuit):
            ml = torch.nn.ModuleList([self]) + other.operations
            return pyq.QuantumCircuit(max(self.n_qubits, other.n_qubits), ml)
        else:
            return TypeError(f"Cannot compose {type(self)} with {type(other)}")

    def __key(self) -> tuple:
        return (self.n_qubits, *self.qubits)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, type(self)):
            return self.__key() == other.__key()
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.__key())

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
