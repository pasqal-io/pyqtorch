from __future__ import annotations

from functools import reduce
from operator import add
from typing import Tuple, Union

import torch
from torch import Tensor, ones_like
from torch.nn import Module, ParameterDict

from pyqtorch.apply import apply_operator
from pyqtorch.circuit import Sequence
from pyqtorch.matrices import _dagger
from pyqtorch.primitive import Primitive
from pyqtorch.utils import Operator, State, is_diag

BATCH_DIM = 2


class Scale(Sequence):
    """Generic container for multiplying a 'Primitive' or 'Sequence' instance by a parameter."""

    def __init__(self, operations: Sequence | Primitive, param_name: str):
        super().__init__(
            operations.operations if isinstance(operations, Sequence) else [operations]
        )
        self.param_name = param_name

    def forward(self, state: Tensor, values: dict[str, Tensor] | ParameterDict = dict()) -> Tensor:
        return (
            values[self.param_name] * super().forward(state, values)
            if isinstance(self.operations, Sequence)
            else self._forward(state, values)
        )

    def _forward(self, state: Tensor, values: dict[str, Tensor]) -> Tensor:
        return apply_operator(state, self.unitary(values), self.operations[0].qubit_support)

    def unitary(self, values: dict[str, Tensor]) -> Tensor:
        thetas = values[self.param_name]
        return thetas * self.operations[0].unitary(values)

    def dagger(self, values: dict[str, Tensor]) -> Tensor:
        return _dagger(self.unitary(values))

    def jacobian(self, values: dict[str, Tensor]) -> Tensor:
        return values[self.param_name] * ones_like(self.unitary(values))


class Add(Sequence):
    """The 'add' operation applies all 'operations' to 'state' and returns the sum of states."""

    def __init__(self, operations: list[Module]):
        super().__init__(operations=operations)

    def forward(self, state: State, values: dict[str, Tensor] | ParameterDict = dict()) -> State:
        return reduce(add, (op(state, values) for op in self.operations))

    def tensor(self, values: dict = dict(), n_qubits: int = 1) -> Tensor:
        if len(self.operations) == 1:
            return self.operations[0].tensor({})
        mat = torch.zeros((2, 2, 1), device=self.device)
        for _ in range(n_qubits - 1):
            mat = torch.kron(mat, torch.zeros((2, 2, 1), device=self.device))
        return reduce(add, (op.tensor(values, n_qubits) for op in self.operations), mat)


TGenerator = Union[torch.nn.ModuleList, list, Tensor, Primitive, None]


class Hamiltonian(Add):
    def __init__(
        self,
        qubit_support: Tuple[int, ...],
        generator: (
            torch.nn.ModuleList | list[Primitive | Sequence] | Tensor | Primitive | None
        ) = [],
    ):
        if isinstance(generator, Tensor):
            generator = [Primitive(generator, target=qubit_support[0])]
        elif generator is None:
            raise NotImplementedError
        super().__init__(operations=generator)
        self.qubit_support = qubit_support


def is_diag_hamiltonian(hamiltonian: Tensor) -> bool:
    diag_check = torch.tensor(
        [is_diag(hamiltonian[..., i]) for i in range(hamiltonian.shape[BATCH_DIM])],
        device=hamiltonian.device,
    )
    return bool(torch.prod(diag_check))


def evolve(hamiltonian: Operator, time_evolution: torch.Tensor) -> Tensor:
    if is_diag_hamiltonian(hamiltonian):
        evol_operator = torch.diagonal(hamiltonian) * (-1j * time_evolution).view((-1, 1))
        evol_operator = torch.diag_embed(torch.exp(evol_operator))
    else:
        evol_operator = torch.transpose(hamiltonian, 0, -1) * (-1j * time_evolution).view(
            (-1, 1, 1)
        )
        evol_operator = torch.linalg.matrix_exp(evol_operator)
    return torch.transpose(evol_operator, 0, -1)


class HamiltonianEvolution(Hamiltonian):
    def __init__(self, qubit_support: Tuple[int, ...], generator: TGenerator, time: Tensor | str):
        super().__init__(qubit_support, generator)
        self.time = time

    def forward(self, state: Tensor, values: dict[str, Tensor] | ParameterDict = dict()) -> Tensor:
        hamiltonian = self.tensor(values, len(self.qubit_support))
        time_evolution = values[self.time] if isinstance(self.time, str) else self.time

        return apply_operator(
            state=state,
            operator=evolve(hamiltonian, time_evolution),
            qubits=self.qubit_support,
            n_qubits=len(state.size()) - 1,
            batch_size=max(hamiltonian.shape[BATCH_DIM], len(time_evolution)),
        )
