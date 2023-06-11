from __future__ import annotations

from typing import Any

import torch
from torch.nn import Module, ModuleList, Parameter, init

from pyqtorch.modules.parametric import Gate
from pyqtorch.modules.primitive import CNOT, PrimitiveGate

PI = 2.0 * torch.asin(torch.Tensor([1.0]).double()).item()


def zero_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> torch.Tensor:
    state = torch.zeros((2**n_qubits, batch_size), dtype=dtype, device=device)
    state[0] = 1
    state = state.reshape([2] * n_qubits + [batch_size])
    return state


def uniform_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> torch.Tensor:
    state = torch.ones((2**n_qubits, batch_size), dtype=dtype, device=device)
    state = state / torch.sqrt(torch.tensor(2**n_qubits))
    state = state.reshape([2] * n_qubits + [batch_size])
    return state


class QuantumCircuit(Module):
    def __init__(self, n_qubits: int, operations: list):
        super().__init__()
        self.n_qubits = n_qubits
        self.operations = torch.nn.ModuleList(operations)

    def __mul__(self, other: PrimitiveGate | Gate | QuantumCircuit) -> QuantumCircuit:
        if self.n_qubits != other.n_qubits:
            raise ValueError(
                (
                    f"Number of Qubits don't match. "
                    f"Left gate is applied on a {self.n_qubits} qubit system and "
                    f"right gate is applied on a {other.n_qubits} qubit system."
                )
            )

        if isinstance(other, QuantumCircuit):
            return QuantumCircuit(self.n_qubits, self.operations.extend(other.operations))

        if isinstance(other, ((Gate, PrimitiveGate))):
            return QuantumCircuit(self.n_qubits, self.operations.append(other))

        else:
            return NotImplemented

    def forward(self, state: torch.Tensor, thetas: torch.Tensor = None) -> torch.Tensor:
        for op in self.operations:
            state = op(state, thetas)
        return state

    @property
    def _device(self) -> torch.device:
        try:
            (_, buffer) = next(self.named_buffers())
            return buffer.device
        except StopIteration:
            return torch.device("cpu")

    def init_state(self, batch_size: int) -> torch.Tensor:
        return zero_state(self.n_qubits, batch_size, device=self._device)

    def matrices(self, thetas: torch.Tensor) -> torch.Tensor:
        matrix = self.operations[0].matrices(thetas[-1]).squeeze(2)
        for i, operation in enumerate(self.operations[1:]):
            matrix = matrix @ operation.matrices(thetas[-2 - i, :]).squeeze(2)
        return matrix.unsqueeze(2)


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
        init.uniform_(self.thetas, -2 * PI, 2 * PI)

    def forward(self, state: torch.Tensor, _: torch.Tensor = None) -> torch.Tensor:
        for (op, t) in zip(self.operations, self.thetas):
            state = op(state, t)
        return state


class EntanglingLayer(QuantumCircuit):
    def __init__(self, n_qubits: int):
        operations = ModuleList(
            [CNOT([i % n_qubits, (i + 1) % n_qubits], n_qubits) for i in range(n_qubits)]
        )
        super().__init__(n_qubits, operations)
