from __future__ import annotations

from typing import Any, Iterator

import torch

from pyqtorch.abstract import AbstractOperator
from pyqtorch.utils import DiffMode, State, overlap, zero_state


class QuantumCircuit(torch.nn.Module):
    def __init__(
        self, n_qubits: int, operations: list[AbstractOperator], diff_mode: DiffMode = DiffMode.AD
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.operations = torch.nn.ModuleList(operations)
        self.diff_mode = diff_mode

    def __mul__(self, other: AbstractOperator | QuantumCircuit) -> QuantumCircuit:
        n_qubits = max(self.n_qubits, other.n_qubits)
        if isinstance(other, QuantumCircuit):
            return QuantumCircuit(n_qubits, self.operations.extend(other.operations))

        elif isinstance(other, AbstractOperator):
            return QuantumCircuit(n_qubits, self.operations.append(other))

        else:
            raise ValueError(f"Cannot compose {type(self)} with {type(other)}")

    def __iter__(self) -> Iterator:
        return iter(self.operations)

    def __key(self) -> tuple:
        return (self.n_qubits,)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, QuantumCircuit):
            return self.__key() == other.__key()
        else:
            raise NotImplementedError(f"Unable to compare QuantumCircuit to {type(other)}.")

    def __hash__(self) -> int:
        return hash(self.__key())

    def run(self, state: State = None, values: dict[str, torch.Tensor] = {}) -> State:
        if state is None:
            state = self.init_state()
        for op in self.operations:
            state = op(state, values)
        return state

    def forward(self, state: State, values: dict[str, torch.Tensor] = {}) -> State:
        return self.run(state, values)

    def expectation(
        self,
        values: dict[str, torch.Tensor],
        observable: QuantumCircuit,
        state: State = None,
    ) -> torch.Tensor:
        if observable is None:
            raise ValueError("Please provide an observable to compute expectation.")
        if state is None:
            state = self.init_state(batch_size=1)
        if self.diff_mode == DiffMode.AD:
            state = self.run(state, values)
            return overlap(state, observable.forward(state, values))
        else:
            from pyqtorch.adjoint import AdjointExpectation

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

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        return zero_state(self.n_qubits, batch_size, device=self._device)

    def reverse(self) -> QuantumCircuit:
        return QuantumCircuit(self.n_qubits, torch.nn.ModuleList(list(reversed(self.operations))))
