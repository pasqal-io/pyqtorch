from __future__ import annotations

import logging
from functools import reduce
from logging import getLogger
from operator import add
from typing import Callable, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, ParameterDict

from pyqtorch.apply import apply_operator
from pyqtorch.circuit import Sequence
from pyqtorch.matrices import _dagger
from pyqtorch.primitive import Primitive
from pyqtorch.utils import (
    State,
    StrEnum,
    inner_prod,
    is_diag,
    operator_to_sparse_diagonal,
)

BATCH_DIM = 2
TGenerator = Union[Tensor, str, Primitive, Sequence]

logger = getLogger(__name__)


def forward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Forward complete")
    torch.cuda.nvtx.range_pop()


def pre_forward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Executing forward")
    torch.cuda.nvtx.range_push("HamiltonianEvolution.forward")


def backward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Backward complete")
    torch.cuda.nvtx.range_pop()


def pre_backward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Executed backward")
    torch.cuda.nvtx.range_push("Hamiltonian Evolution.backward")


class GeneratorType(StrEnum):
    """
    Options for types of generators allowed in HamiltonianEvolution.
    """

    PARAMETRIC_OPERATION = "parametric_operation"
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
        self.param_name = param_name
        assert len(self.operations) == 1

    def forward(
        self, state: Tensor, values: dict[str, Tensor] | ParameterDict = dict()
    ) -> Tensor:
        return (
            values[self.param_name] * super().forward(state, values)
            if isinstance(self.operations, Sequence)
            else self._forward(state, values)
        )

    def _forward(self, state: Tensor, values: dict[str, Tensor]) -> Tensor:
        return apply_operator(
            state, self.unitary(values), self.operations[0].qubit_support
        )

    def unitary(self, values: dict[str, Tensor]) -> Tensor:
        thetas = (
            values[self.param_name]
            if isinstance(self.param_name, str)
            else self.param_name
        )
        return thetas * self.operations[0].unitary(values)

    def dagger(self, values: dict[str, Tensor]) -> Tensor:
        return _dagger(self.unitary(values))

    def jacobian(self, values: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError(
            "The Jacobian of `Scale` is done via decomposing it into the gradient w.r.t\
                                  the scale parameter and the gradient w.r.t to the scaled block."
        )
        # TODO make scale a primitive block with an additional parameter
        # So you can do the following:
        # thetas = values[self.param] if isinstance(self.param, str) else self.param_name
        # return thetas * ones_like(self.unitary(values))

    def tensor(
        self,
        values: dict[str, Tensor] = dict(),
        n_qubits: int | None = None,
        diagonal: bool = False,
    ) -> Tensor:
        thetas = (
            values[self.param_name]
            if isinstance(self.param_name, str)
            else self.param_name
        )
        return thetas * self.operations[0].tensor(values, n_qubits, diagonal)

    def flatten(self) -> list[Scale]:
        return [self]  # This method should only be called in the AdjointExpectation,
        # where the `Scale` is only supported for Primitive (and not Sequences)
        # so we don't want to flatten this to preserve the scale parameter


class Add(Sequence):
    """The 'add' operation applies all 'operations' to 'state' and returns the sum of states."""

    def __init__(self, operations: list[Module]):
        super().__init__(operations=operations)

    def forward(
        self, state: State, values: dict[str, Tensor] | ParameterDict = dict()
    ) -> State:
        return reduce(add, (op(state, values) for op in self.operations))

    def tensor(
        self, values: dict = {}, n_qubits: int | None = None, diagonal: bool = False
    ) -> Tensor:
        if n_qubits is None:
            n_qubits = max(self.qubit_support) + 1
        mat = torch.zeros((2, 2, 1), device=self.device)
        for _ in range(n_qubits - 1):
            mat = torch.kron(mat, torch.zeros((2, 2, 1), device=self.device))
        return reduce(
            add, (op.tensor(values, n_qubits, diagonal) for op in self.operations), mat
        )


class Observable(Sequence):
    def __init__(
        self,
        n_qubits: int | None,
        operations: list[Module] | Primitive | Sequence,
    ):
        super().__init__(operations)
        if n_qubits is None:
            n_qubits = max(self.qubit_support) + 1
        self.n_qubits = n_qubits

    def run(self, state: Tensor, values: dict[str, Tensor]) -> Tensor:
        for op in self.operations:
            state = op(state, values)
        return state

    def forward(
        self, state: Tensor, values: dict[str, Tensor] | ParameterDict = dict()
    ) -> Tensor:
        return inner_prod(state, self.run(state, values)).real


class DiagonalObservable(Primitive):
    def __init__(
        self,
        n_qubits: int | None,
        operations: list[Module] | Primitive | Sequence,
        to_sparse: bool = False,
    ):
        """In case the 'operations' / hamiltonian is diagonal,
        we simply do a element-wise vector-product instead of a tensordot."""
        if isinstance(operations, list):
            operations = Sequence(operations)
        if n_qubits is None:
            n_qubits = max(operations.qubit_support) + 1
        hamiltonian = operations.tensor({}, n_qubits).squeeze(2)
        if to_sparse:
            operator = operator_to_sparse_diagonal(hamiltonian)
        else:
            operator = torch.diag(hamiltonian).reshape(-1, 1)
        super().__init__(operator, operations.qubit_support[0])
        self.qubit_support = operations.qubit_support
        self.n_qubits = n_qubits

    def run(self, state: Tensor, values: dict[str, Tensor]) -> Tensor:
        # We flatten the state, do a element-wise multiplication with the diagonal hamiltonian
        # and reshape it back to pyq-shape.
        return torch.einsum(
            "ij,ib->ib", self.pauli, state.flatten(start_dim=0, end_dim=-2)
        ).reshape([2] * self.n_qubits + [state.shape[-1]])

    def forward(
        self, state: Tensor, values: dict[str, Tensor] | ParameterDict = dict()
    ) -> Tensor:
        return inner_prod(state, self.run(state, values)).real


def is_diag_hamiltonian(hamiltonian: Tensor) -> bool:
    diag_check = torch.tensor(
        [is_diag(hamiltonian[..., i]) for i in range(hamiltonian.shape[BATCH_DIM])],
        device=hamiltonian.device,
    )
    return bool(torch.prod(diag_check))


def evolve(hamiltonian: Tensor, time_evolution: Tensor) -> Tensor:
    if is_diag_hamiltonian(hamiltonian):
        evol_operator = torch.diagonal(hamiltonian) * (-1j * time_evolution).view(
            (-1, 1)
        )
        evol_operator = torch.diag_embed(torch.exp(evol_operator))
    else:
        evol_operator = torch.transpose(hamiltonian, 0, -1) * (
            -1j * time_evolution
        ).view((-1, 1, 1))
        evol_operator = torch.linalg.matrix_exp(evol_operator)
    return torch.transpose(evol_operator, 0, -1)


class HamiltonianEvolution(Sequence):
    def __init__(
        self,
        generator: TGenerator,
        time: Tensor | str,
        qubit_support: Tuple[int, ...] | None = None,
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
            ), "When using a symbolic generator, please pass a qubit_support."
            self.generator_type = GeneratorType.SYMBOL
            self.generator_symbol = generator
            generator = []
        elif isinstance(generator, (Primitive, Sequence)):
            qubit_support = generator.qubit_support
            if generator_parametric:
                generator = [generator]
                self.generator_type = GeneratorType.PARAMETRIC_OPERATION
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
        self._qubit_support = qubit_support  # type: ignore
        self.time = time
        logger.debug("Hamiltonian Evolution initialized")
        if logger.isEnabledFor(logging.DEBUG):
            # When Debugging let's add logging and NVTX markers
            # WARNING: incurs performance penalty
            self.register_forward_hook(forward_hook, always_call=True)
            self.register_full_backward_hook(backward_hook)
            self.register_forward_pre_hook(pre_forward_hook)
            self.register_full_backward_pre_hook(pre_backward_hook)

        self._generator_map: dict[GeneratorType, Callable[[dict], Tensor]] = {
            GeneratorType.SYMBOL: self._symbolic_generator,
            GeneratorType.TENSOR: self._tensor_generator,
            GeneratorType.OPERATION: self._tensor_generator,
            GeneratorType.PARAMETRIC_OPERATION: self._parametric_generator,
        }

    @property
    def generator(self) -> ModuleList:
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

    def forward(
        self, state: Tensor, values: dict[str, Tensor] | ParameterDict = dict()
    ) -> Tensor:
        hamiltonian: torch.Tensor = self.create_hamiltonian(values)
        time_evolution: torch.Tensor = (
            values[self.time] if isinstance(self.time, str) else self.time
        )  # If `self.time` is a string / hence, a Parameter,
        # we expect the user to pass it in the `values` dict

        return apply_operator(
            state=state,
            operator=evolve(hamiltonian, time_evolution),
            qubits=self.qubit_support,
            n_qubits=len(state.size()) - 1,
            batch_size=max(hamiltonian.shape[BATCH_DIM], len(time_evolution)),
        )

    def tensor(
        self,
        values: dict = {},
        n_qubits: int | None = None,
        diagonal: bool = False,
    ) -> Tensor:
        hamiltonian: torch.Tensor = self.create_hamiltonian(values)
        time_evolution: torch.Tensor = (
            values[self.time] if isinstance(self.time, str) else self.time
        )
        return evolve(hamiltonian, time_evolution)
