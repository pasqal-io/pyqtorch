from __future__ import annotations

from logging import getLogger
from typing import Any, Iterator

from torch import Tensor, device
from torch.nn import Module, ModuleList

from pyqtorch.utils import DiffMode, State, overlap, zero_state

logger = getLogger(__name__)


class QuantumCircuit(Module):
    def __init__(self, n_qubits: int, operations: list[Module]):
        super().__init__()
        self.n_qubits = n_qubits
        self.operations = ModuleList(operations)

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

    @property
    def _device(self) -> device | None:
        devices = set()
        for op in self.operations:
            if isinstance(op, QuantumCircuit):
                devices.add(op._device)
            elif isinstance(op, Module):
                devices.update([b.device for b in op.buffers()])
        if len(devices) == 1 and None not in devices:
            _device = next(iter(devices))
            logger.debug(f"Found device {_device}.")
            return _device
        else:
            logger.warning(
                f"Unable to determine device of module {self}.\
                             Found {devices}, however expected exactly one device."
            )
            return None

    def init_state(self, batch_size: int = 1) -> Tensor:
        return zero_state(self.n_qubits, batch_size, device=self._device)

    def reverse(self) -> QuantumCircuit:
        return QuantumCircuit(self.n_qubits, ModuleList(list(reversed(self.operations))))

    def to(self, device: device) -> QuantumCircuit:
        self.operations = ModuleList([op.to(device) for op in self.operations])
        return self


def expectation(
    circuit: QuantumCircuit,
    state: State,
    values: dict[str, Tensor],
    observable: QuantumCircuit,
    diff_mode: DiffMode = DiffMode.AD,
) -> Tensor:
    """Compute the expectation value of the circuit given a state and observable.
    Arguments:
        circuit: QuantumCircuit instance
        state: An input state
        values: A dictionary of parameter values
        observable: QuantumCircuit representing the observable
        diff_mode: The differentiation mode
    Returns:
        A expectation value.
    """
    if observable is None:
        raise ValueError("Please provide an observable to compute expectation.")
    if state is None:
        state = circuit.init_state(batch_size=1)
    if diff_mode == DiffMode.AD:
        state = circuit.run(state, values)
        return overlap(state, observable.forward(state, values))
    elif diff_mode == DiffMode.ADJOINT:
        from pyqtorch.adjoint import AdjointExpectation

        return AdjointExpectation.apply(circuit, observable, state, values.keys(), *values.values())
    else:
        raise ValueError(f"Requested diff_mode '{diff_mode}' not supported.")
