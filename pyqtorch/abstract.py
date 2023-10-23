from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.nn import Module

import pyqtorch as pyq
from pyqtorch.utils import Operator, State


class AbstractOperator(ABC, Module):
    def __init__(self, target: int):
        super().__init__()
        self.target: int = target
        self.qubit_support: list[int] = [target]
        self.n_qubits: int = len(self.qubit_support)

    def __mul__(self, other: AbstractOperator | pyq.QuantumCircuit) -> pyq.QuantumCircuit:
        if isinstance(other, AbstractOperator):
            ops = torch.nn.ModuleList([self, other])
            return pyq.QuantumCircuit(max(self.target, other.target), ops)
        elif isinstance(other, pyq.QuantumCircuit):
            ops = torch.nn.ModuleList([self]) + other.operations
            return pyq.QuantumCircuit(max(self.target, other.target), ops)
        else:
            raise TypeError(f"Unable to compose {type(self)} with {type(other)}")

    def __key(self) -> tuple:
        return (self.target,)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, type(self)):
            return self.__key() == other.__key()
        else:
            raise NotImplementedError

    def __hash__(self) -> int:
        return hash(self.__key())

    @abstractmethod
    def unitary(self, values: dict[str, torch.Tensor]) -> Operator:
        ...

    @abstractmethod
    def dagger(self, values: dict[str, torch.Tensor]) -> Operator:
        ...

    @abstractmethod
    def forward(self, state: torch.Tensor, values: dict[str, torch.Tensor]) -> State:
        ...

    def extra_repr(self) -> str:
        return f"qubits={self.target}"
