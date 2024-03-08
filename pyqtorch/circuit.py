from __future__ import annotations

import logging
from logging import getLogger
from typing import Any, Iterator

import torch
from torch import Tensor
from torch import device as torch_device
from torch.nn import Module, ModuleList

from pyqtorch.utils import DiffMode, State, inner_prod, zero_state

logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)


def forward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Forward complete")
    torch.cuda.nvtx.range_pop()


def pre_forward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Executing forward")
    torch.cuda.nvtx.range_push("QuantumCircuit.forward")


def backward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Backward complete")
    torch.cuda.nvtx.range_pop()


def pre_backward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Executed backward")
    torch.cuda.nvtx.range_push("QuantumCircuit.backward")


class QuantumCircuit(Module):
    def __init__(self, n_qubits: int, operations: list[Module]):
        super().__init__()
        self.n_qubits = n_qubits
        self.operations = ModuleList(operations)
        self._device = torch_device("cpu")
        if operations:
            try:
                self._device = next(iter(set((op.device for op in self.operations))))
            except StopIteration:
                pass
        logger.debug("QuantumCircuit initialized")
        if logger.isEnabledFor(logging.DEBUG):
            # When Debugging let's add logging and NVTX markers
            # WARNING: incurs performance penalty
            self.register_forward_hook(forward_hook, always_call=True)
            self.register_full_backward_hook(backward_hook)
            self.register_forward_pre_hook(pre_forward_hook)
            self.register_full_backward_pre_hook(pre_backward_hook)

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

    def forward(self, state: State, values: dict[str, Tensor] = {}) -> State:
        for op in self.operations:
            state = op(state, values)
        return state

    def run(self, state: State = None, values: dict[str, Tensor] = {}) -> State:
        if state is None:
            state = self.init_state()
        return self.forward(state, values)

    @property
    def device(self) -> torch_device:
        return self._device

    def init_state(self, batch_size: int = 1) -> Tensor:
        return zero_state(self.n_qubits, batch_size, device=self.device)

    def reverse(self) -> QuantumCircuit:
        return QuantumCircuit(self.n_qubits, ModuleList(list(reversed(self.operations))))

    def to(self, device: torch_device) -> QuantumCircuit:
        self.operations = ModuleList([op.to(device) for op in self.operations])
        self._device = device
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
    if diff_mode == DiffMode.AD:
        state = circuit.forward(state, values)
        return inner_prod(state, observable.forward(state, values)).real
    elif diff_mode == DiffMode.ADJOINT:
        from pyqtorch.adjoint import AdjointExpectation

        if state is None:
            state = circuit.init_state(batch_size=1)
        return AdjointExpectation.apply(circuit, observable, state, values.keys(), *values.values())
    else:
        raise ValueError(f"Requested diff_mode '{diff_mode}' not supported.")
