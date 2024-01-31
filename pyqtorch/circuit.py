from __future__ import annotations

from logging import getLogger
from typing import Any, Iterator

from torch import Tensor, device
from torch.nn import Module, ModuleList

from pyqtorch.utils import DiffMode, State, overlap, zero_state

logger = getLogger(__name__)


class QuantumCircuit(Module):
    def __init__(self, n_qubits: int, operations: list[Module], diff_mode: DiffMode = DiffMode.AD):
        super().__init__()
        self.n_qubits = n_qubits
        self.operations = ModuleList(operations)
        self.diff_mode = diff_mode

    def __mul__(self, other: Module | QuantumCircuit) -> QuantumCircuit:
        n_qubits = max(self.n_qubits, other.n_qubits)
        if isinstance(other, QuantumCircuit):
            return QuantumCircuit(n_qubits, self.operations.extend(other.operations))

        elif isinstance(other, Module):
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

    def run(self, state: State = None, values: dict[str, Tensor] = {}) -> State:
        if state is None:
            state = self.init_state()
        for op in self.operations:
            state = op(state, values)
        return state

    def forward(self, state: State, values: dict[str, Tensor] = {}) -> State:
        return self.run(state, values)

    def expectation(
        self,
        values: dict[str, Tensor],
        observable: QuantumCircuit,
        state: State = None,
    ) -> Tensor:
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
    def _device(self) -> device | None:
        device_lst: list[device] = []
        for op in self.operations:
            if isinstance(op, QuantumCircuit):
                device_lst.append(op._device)
            elif isinstance(op, Module):
                device_lst += [b.device for b in op.buffers()]
        if len(list(set(device_lst))) > 1:
            logger.error(f"Found more than one device in object {self}.")
            return None
        elif len(device_lst) == 0:
            logger.warning(f"Unable to determine device on module {self}.")
            return None
        else:
            return device_lst[0]

    def init_state(self, batch_size: int = 1) -> Tensor:
        return zero_state(self.n_qubits, batch_size, device=self._device)

    def reverse(self) -> QuantumCircuit:
        return QuantumCircuit(self.n_qubits, ModuleList(list(reversed(self.operations))))

    def to(self, device: device) -> QuantumCircuit:
        self.operations = ModuleList([op.to(device) for op in self.operations])
        return self
