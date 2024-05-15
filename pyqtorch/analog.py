from __future__ import annotations

from functools import reduce
from operator import add
from typing import Tuple

import torch
from torch import Tensor, ones_like
from torch.nn import Module, ParameterDict

from pyqtorch.apply import apply_operator
from pyqtorch.circuit import Sequence
from pyqtorch.matrices import _dagger
from pyqtorch.parametric import Parametric
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


class Hamiltonian(Add):
    def __init__(self, operations: list[torch.nn.Module]):
        if all([not isinstance(op, (Parametric)) for op in operations]):
            super().__init__(operations)
        else:
            raise TypeError(
                "Hamiltonian can only contain the following operations: [Primitive, Scale, Add]."
            )


class HamiltonianEvolution(torch.nn.Module):
    def __init__(
        self,
        qubit_support: Tuple[int, ...],
    ):
        super().__init__()
        self.qubit_support: Tuple[int, ...] = qubit_support

        def _diag_operator(hamiltonian: Operator, time_evolution: torch.Tensor) -> Operator:
            evol_operator = torch.diagonal(hamiltonian) * (-1j * time_evolution).view((-1, 1))
            evol_operator = torch.diag_embed(torch.exp(evol_operator))
            return torch.transpose(evol_operator, 0, -1)

        def _matrixexp_operator(hamiltonian: Operator, time_evolution: torch.Tensor) -> Operator:
            evol_operator = torch.transpose(hamiltonian, 0, -1) * (-1j * time_evolution).view(
                (-1, 1, 1)
            )
            evol_operator = torch.linalg.matrix_exp(evol_operator)
            return torch.transpose(evol_operator, 0, -1)

        self._evolve_diag_operator = _diag_operator
        self._evolve_matrixexp_operator = _matrixexp_operator

    def forward(
        self,
        hamiltonian: torch.Tensor,
        time_evolution: torch.Tensor,
        state: State,
    ) -> State:
        if len(hamiltonian.size()) < 3:
            hamiltonian = hamiltonian.unsqueeze(2)
        batch_size = max(hamiltonian.shape[BATCH_DIM], len(time_evolution))
        diag_check = torch.tensor(
            [is_diag(hamiltonian[..., i]) for i in range(hamiltonian.shape[BATCH_DIM])],
            device=hamiltonian.device,
        )
        evolve_operator = (
            self._evolve_diag_operator
            if bool(torch.prod(diag_check))
            else self._evolve_matrixexp_operator
        )
        return apply_operator(
            state,
            evolve_operator(hamiltonian, time_evolution),
            self.qubit_support,
            len(state.size()) - 1,
            batch_size,
        )
