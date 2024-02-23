from __future__ import annotations

from logging import getLogger
from typing import Any, Iterator

import torch
from torch import Tensor
from torch import device as torch_device
from torch.nn import Module, ModuleList

from pyqtorch.utils import DiffMode, State, inner_prod, zero_state

logger = getLogger(__name__)


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
    if state is None:
        state = circuit.init_state(batch_size=1)
    if diff_mode == DiffMode.AD:
        state = circuit.run(state, values)
        return inner_prod(state, observable.forward(state, values)).real
    elif diff_mode == DiffMode.ADJOINT:
        from pyqtorch.adjoint import AdjointExpectation

        return AdjointExpectation.apply(circuit, observable, state, values.keys(), *values.values())
    else:
        raise ValueError(f"Requested diff_mode '{diff_mode}' not supported.")


class PipedCircuit(QuantumCircuit):
    def __init__(self, n_qubits: int, operations: list[Module], dev_idx: int):
        super().__init__(n_qubits, operations)
        self = self.to(torch_device(f"cuda:{dev_idx}"))

    def run(self, state: State = None, values: dict[str, Tensor] = {}) -> State:
        if state is None:
            state = self.init_state()
        else:
            state = state.to(self.device)
        values = {k: v.to(self.device) for k, v in values.items()}
        for op in self.operations:
            state = op(state, values)
        return state


class ModelParallelCircuit(QuantumCircuit):
    def __init__(self, circ: QuantumCircuit, n_devices: int):
        if not all([isinstance(subc, QuantumCircuit) for subc in circ.operations]):
            msg = "Make sure the passed QuantumCircuit only contains other QuantumCircuits."
            logger.error(msg)
            raise ValueError(msg)
        if not torch.cuda.is_available():
            msg = f"{self.__class__.__name__} requires at least two GPU devices."
            logger.error(msg)
            raise ValueError(msg)
        dev_count = torch.cuda.device_count()
        if dev_count < n_devices:
            msg = f"Requested {n_devices} GPU devices however only {dev_count} devices available."
            logger.error(msg)
            raise ValueError(msg)
        n_circs = len(circ.operations)
        dev_indices = [i for i in range(n_devices) for _ in range(n_circs // n_devices)]
        operations = [
            PipedCircuit(c.n_qubits, c.operations, dev_idx)
            for c, dev_idx in zip(circ.operations, dev_indices)
        ]
        super().__init__(circ.n_qubits, operations)


# class PipelineParallelCircuit(ModelParallelCircuit):
#     def __init__(self, split_size=N_POINTS // 2, *args, **kwargs):
#         super(PipelineParallelCircuit, self).__init__(*args, **kwargs)
#         self.split_size = split_size

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         init_state = zero_state(N_QUBITS).to("cuda:0")
#         splits = iter(x.split(self.split_size, dim=0))
#         s_next = next(splits)
#         s_prev = self.feature_map(init_state, {"x": s_next.to("cuda:0")})
#         s_prev = self.c0.forward(s_prev, self.params_c0).to("cuda:1")
#         ret = []

#         for s_next in splits:
#             s_prev = self.c1.forward(s_prev, self.params_c1)
#             ret.append(inner_prod(s_prev, self.observable.forward(s_prev)).real)

#             s_prev = self.feature_map(init_state, {"x": s_next.to("cuda:0")})
#             s_prev = self.c0.forward(s_prev, self.params_c0).to("cuda:1")

#         s_prev = self.c1.forward(s_prev, self.params_c1)
#         ret.append(inner_prod(s_prev, self.observable.forward(s_prev)).real)
#         return torch.cat(ret)


# def train(circ) -> None:
#     optimizer = torch.optim.Adam(
#         {**circ.params_c0, **circ.params_c1}.values(), lr=0.01, foreach=False
#     )
#     for epoch in range(N_EPOCHS):
#         optimizer.zero_grad()
#         y_pred = circ.forward(x)
#         loss = mse_loss(y_pred, y.to("cuda:1"))
#         loss.backward()
#         optimizer.step()
