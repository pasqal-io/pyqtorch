from __future__ import annotations

from typing import Any

import torch
from torch.nn import Module, ModuleList, Parameter, init

from pyqtorch.adjoint import AdjointExpectation
from pyqtorch.operator import Operator
from pyqtorch.primitive import CNOT
from pyqtorch.utils import DiffMode, overlap, zero_state


class QuantumCircuit(Module):
    def __init__(
        self, n_qubits: int, operations: list[Operator], diff_mode: DiffMode = DiffMode.AD
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.operations = torch.nn.ModuleList(operations)
        self.diff_mode = diff_mode

    def __mul__(self, other: Operator | QuantumCircuit) -> QuantumCircuit:
        if isinstance(other, QuantumCircuit):
            n_qubits = max(self.n_qubits, other.n_qubits)
            return QuantumCircuit(n_qubits, self.operations.extend(other.operations))

        if isinstance(other, Operator):
            n_qubits = max(self.n_qubits, other.n_qubits)
            return QuantumCircuit(n_qubits, self.operations.append(other))

        else:
            raise ValueError(f"Cannot compose {type(self)} with {type(other)}")

    def __key(self) -> tuple:
        return (self.n_qubits, *self.operations)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, QuantumCircuit):
            return self.__key() == other.__key()
        raise NotImplementedError

    def __hash__(self) -> int:
        return hash(self.__key())

    def forward(self, state: torch.Tensor, values: dict[str, torch.Tensor] = {}) -> torch.Tensor:
        for op in self.operations:
            state = op(state, values)
        return state

    def run(self, state: torch.Tensor, values: dict[str, torch.Tensor] = {}) -> torch.Tensor:
        return self.forward(state, values)

    def expectation(
        self,
        values: dict[str, torch.Tensor] = {},
        observable: QuantumCircuit = None,
        state: torch.Tensor = None,
    ) -> torch.Tensor:
        if state is None:
            state = self.init_state(batch_size=1)
        if observable is None:
            raise ValueError("Please provide an observable to compute expectation.")
        if self.diff_mode == DiffMode.AD:
            state = self.run(state, values)
            return overlap(state, observable.forward(state, values))
        else:
            return AdjointExpectation.apply(
                self, observable, state, values.keys(), *values.values()
            )

    @property
    def _device(self) -> torch.device:
        try:
            (_, buffer) = next(self.named_buffers())
            return buffer.device
        except StopIteration:
            return torch.device("cpu")

    def init_state(self, batch_size: int) -> torch.Tensor:
        return zero_state(self.n_qubits, batch_size, device=self._device)

    def reverse(self) -> QuantumCircuit:
        return QuantumCircuit(self.n_qubits, torch.nn.ModuleList(list(reversed(self.operations))))


def FeaturemapLayer(n_qubits: int, Op: Any) -> QuantumCircuit:
    operations = [Op([i], n_qubits) for i in range(n_qubits)]
    return QuantumCircuit(n_qubits, operations)


class VariationalLayer(QuantumCircuit):
    def __init__(self, n_qubits: int, Op: Any):
        operations = ModuleList([Op([i], n_qubits) for i in range(n_qubits)])
        super().__init__(n_qubits, operations)

        self.thetas = Parameter(torch.empty(n_qubits, Op.n_params))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.uniform_(self.thetas, -2 * torch.pi, 2 * torch.pi)

    def forward(self, state: torch.Tensor, _: torch.Tensor = None) -> torch.Tensor:
        for op, t in zip(self.operations, self.thetas):
            state = op(state, t)
        return state


class EntanglingLayer(QuantumCircuit):
    def __init__(self, n_qubits: int):
        operations = ModuleList(
            [CNOT([i % n_qubits, (i + 1) % n_qubits], n_qubits) for i in range(n_qubits)]
        )
        super().__init__(n_qubits, operations)
