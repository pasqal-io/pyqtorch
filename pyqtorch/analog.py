from __future__ import annotations

from functools import reduce
from operator import add
from typing import Callable, Tuple, Union

import torch
from torch import Tensor, ones_like
from torch.nn import Module, ParameterDict

from pyqtorch.apply import apply_operator
from pyqtorch.circuit import Sequence
from pyqtorch.matrices import _dagger
from pyqtorch.primitive import Primitive
from pyqtorch.utils import Operator, State, StrEnum, is_diag

BATCH_DIM = 2


class GeneratorType(StrEnum):
    """
    Options for types of generators allowed in HamiltonianEvolution.
    """

    OPERATION_PARAMETRIC = "operation_parametric"
    """Generators of type Primitive or Sequence which contain
       possibly trainable or non-trainable parameters."""
    OPERATION = "operation"
    """Generators of type Primitive or Sequence which do not contain parameters or contain
       constants as Parameters for example pyq.Scale(Z(0), torch.tensor([1.]))."""
    TENSOR = "tensor"
    """Generators of type torch.Tensor in which case a qubit_support needs to be passed."""
    SYMBOL = "symbol"
    """Generators which are symbolic, i.e. will be passed via the 'values' dict by the user."""


class Scale(Sequence):
    """Generic container for multiplying a 'Primitive' or 'Sequence' instance by a parameter."""

    def __init__(self, operations: list[Module], param_name: str | Tensor):
        super().__init__(
            operations.operations if isinstance(operations, Sequence) else [operations]
        )
        self.param = param_name
        assert len(self.operations) == 1

    def forward(self, state: Tensor, values: dict[str, Tensor] | ParameterDict = dict()) -> Tensor:
        return (
            values[self.param] * super().forward(state, values)
            if isinstance(self.operations, Sequence)
            else self._forward(state, values)
        )

    def _forward(self, state: Tensor, values: dict[str, Tensor]) -> Tensor:
        return apply_operator(state, self.unitary(values), self.operations[0].qubit_support)

    def unitary(self, values: dict[str, Tensor]) -> Tensor:
        thetas = values[self.param] if isinstance(self.param, str) else self.param
        return thetas * self.operations[0].unitary(values)

    def dagger(self, values: dict[str, Tensor]) -> Tensor:
        return _dagger(self.unitary(values))

    def jacobian(self, values: dict[str, Tensor]) -> Tensor:
        thetas = values[self.param] if isinstance(self.param, str) else self.param
        return thetas * ones_like(self.unitary(values))

    def tensor(self, values: dict[str, Tensor] = {}, n_qubits: int = 1) -> Tensor:
        thetas = values[self.param] if isinstance(self.param, str) else self.param
        return thetas * self.operations[0].tensor(values, n_qubits)


class Add(Sequence):
    """The 'add' operation applies all 'operations' to 'state' and returns the sum of states."""

    def __init__(self, operations: list[Module]):
        super().__init__(operations=operations)

    def forward(self, state: State, values: dict[str, Tensor] | ParameterDict = dict()) -> State:
        return reduce(add, (op(state, values) for op in self.operations))

    def tensor(self, values: dict = {}, n_qubits: int = 1) -> Tensor:
        mat = torch.zeros((2, 2, 1), device=self.device)
        for _ in range(n_qubits - 1):
            mat = torch.kron(mat, torch.zeros((2, 2, 1), device=self.device))
        return reduce(add, (op.tensor(values, n_qubits) for op in self.operations), mat)


TGenerator = Union[torch.nn.ModuleList, list, Tensor, Primitive]


class Hamiltonian(Add):
    def __init__(
        self,
        generator: TGenerator,
        qubit_support: Tuple[int, ...],
    ):
        assert isinstance(
            generator, (Tensor, Primitive, Sequence)
        ), "Generator can be: primitive, tensor or sequence"
        if isinstance(generator, Tensor):
            generator = [Primitive(generator, target=qubit_support[0])]
        elif isinstance(generator, Primitive):
            generator = [generator]
        super().__init__(operations=generator)
        self.qubit_support = qubit_support
        self.generator = self.operations


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


class HamiltonianEvolution(Sequence):
    def __init__(
        self,
        generator: TGenerator,
        time: Tensor | str,
        qubit_support: Tuple[int, ...] = None,
        generator_parametric: bool = False,
    ):
        if isinstance(generator, Tensor):
            assert (
                qubit_support is not None
            ), "When using a Tensor generator, please pass a qubit_support."
            if len(generator.shape) < 3:
                generator = generator.unsqueeze(2)
            generator = [Primitive(generator, target=-1)]
            self.generator_type = GeneratorType.TENSOR
        elif isinstance(generator, str):
            assert (
                qubit_support is not None
            ), "When using a Tensor generator, please pass a qubit_support."
            self.generator_type = GeneratorType.SYMBOL
            self.generator_symbol = generator
            generator = []
        elif isinstance(generator, (Primitive, Sequence)):
            qubit_support = generator.qubit_support
            if generator_parametric:
                generator = [generator]
                self.generator_type = GeneratorType.OPERATION_PARAMETRIC
            else:
                generator = [
                    Primitive(
                        generator.tensor({}, len(generator.qubit_support)),
                        target=generator.qubit_support[0],
                    )
                ]
                self.generator_type = GeneratorType.OPERATION
        else:
            raise TypeError(
                f"Received generator of type {type(generator)},\
                            allowed types are: [Tensor, str, Primitive, Sequence]"
            )
        super().__init__(generator)
        self.qubit_support = qubit_support  # type: ignore
        self.time = time

        self._generator_map: dict[GeneratorType, Callable[[dict], Tensor]] = {
            GeneratorType.SYMBOL: self._symbolic_generator,
            GeneratorType.TENSOR: self._tensor_generator,
            GeneratorType.OPERATION: self._tensor_generator,
            GeneratorType.OPERATION_PARAMETRIC: self._parametric_generator,
        }

    @property
    def generator(self) -> TGenerator:
        return self.operations

    def _symbolic_generator(self, values: dict) -> Tensor:
        hamiltonian = values[self.generator_symbol]
        return hamiltonian.unsqueeze(2) if len(hamiltonian.shape) == 2 else hamiltonian

    def _tensor_generator(self, values: dict = {}) -> Tensor:
        return self.generator[0].pauli

    def _parametric_generator(self, values: dict) -> Tensor:
        return self.generator[0].tensor(values, len(self.qubit_support))

    @property
    def create_hamiltonian(self) -> Callable[[dict], Tensor]:
        return self._generator_map[self.generator_type]

    def forward(self, state: Tensor, values: dict[str, Tensor] | ParameterDict = dict()) -> Tensor:
        hamiltonian = self.create_hamiltonian(values)
        time_evolution = values[self.time] if isinstance(self.time, str) else self.time

        return apply_operator(
            state=state,
            operator=evolve(hamiltonian, time_evolution),
            qubits=self.qubit_support,
            n_qubits=len(state.size()) - 1,
            batch_size=max(hamiltonian.shape[BATCH_DIM], len(time_evolution)),
        )
