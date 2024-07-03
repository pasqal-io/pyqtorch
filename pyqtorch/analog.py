from __future__ import annotations

import logging
from functools import reduce
from logging import getLogger
from operator import add
from typing import Any, Callable, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, ParameterDict

from pyqtorch.apply import apply_operator
from pyqtorch.circuit import Sequence
from pyqtorch.matrices import _dagger
from pyqtorch.primitive import Primitive
from pyqtorch.utils import (
    ATOL,
    Operator,
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
    """
    Generic container for multiplying a 'Primitive' or 'Sequence' instance by a parameter.

    Attributes:
        operations: Operations making the Sequence.
        param_name: Name of the parameter to multiply operations with.
    """

    def __init__(self, operations: Union[Sequence, Module], param_name: str | Tensor):
        """
        Initializes a Scale object.

        Arguments:
            operations: Operations making the Sequence.
            param_name: Name of the parameter to multiply operations with.
        """
        super().__init__(
            operations.operations if isinstance(operations, Sequence) else [operations]
        )
        self.param_name = param_name
        assert len(self.operations) == 1

    def forward(
        self, state: Tensor, values: dict[str, Tensor] | ParameterDict = dict()
    ) -> State:
        """
        Apply the operation(s) multiplying by the parameter value.

        Arguments:
            state: Input state.
            values: Parameter value.

        Returns:
            The transformed state.
        """
        return (
            values[self.param_name] * super().forward(state, values)
            if isinstance(self.operations, Sequence)
            else self._forward(state, values)
        )

    def _forward(self, state: Tensor, values: dict[str, Tensor]) -> State:
        """
        Apply the single operation of Scale multiplied by the parameter value.

        Arguments:
            state: Input state.
            values: Parameter value.

        Returns:
            The transformed state.
        """
        return apply_operator(
            state, self.unitary(values), self.operations[0].qubit_support
        )

    def unitary(self, values: dict[str, Tensor]) -> Operator:
        """
        Get the corresponding unitary.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation.
        """
        thetas = (
            values[self.param_name]
            if isinstance(self.param_name, str)
            else self.param_name
        )
        return thetas * self.operations[0].unitary(values)

    def dagger(self, values: dict[str, Tensor]) -> Operator:
        """
        Get the corresponding unitary of the dagger.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation of the dagger.
        """
        return _dagger(self.unitary(values))

    def jacobian(self, values: dict[str, Tensor]) -> Operator:
        """
        Get the corresponding unitary of the jacobian.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation of the jacobian.

        Raises:
            NotImplementedError
        """
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
    ) -> Operator:
        """
        Get the corresponding unitary over n_qubits.

        Arguments:
            values: Parameter value.
            n_qubits: The number of qubits the unitary is represented over.
            Can be higher than the number of qubit support.
            diagonal: Whether the operation is diagonal.


        Returns:
            The unitary representation.
        Raises:
            NotImplementedError for the diagonal case.
        """
        thetas = (
            values[self.param_name]
            if isinstance(self.param_name, str)
            else self.param_name
        )
        return thetas * self.operations[0].tensor(values, n_qubits, diagonal)

    def flatten(self) -> list[Scale]:
        """This method should only be called in the AdjointExpectation,
        where the `Scale` is only supported for Primitive (and not Sequences)
        so we don't want to flatten this to preserve the scale parameter.

        Returns:
            The Scale within a list.
        """
        return [self]

    def to(self, *args: Any, **kwargs: Any) -> Scale:
        """Perform conversions for dtype or device.

        Returns:
            Converted Scale.
        """
        super().to(*args, **kwargs)
        if not isinstance(self.param_name, str):
            self.param_name = self.param_name.to(*args, **kwargs)

        return self


class Add(Sequence):
    """
    The 'add' operation applies all 'operations' to 'state' and returns the sum of states.

    Attributes:
        operations: List of operations to add up.
    """

    def __init__(self, operations: list[Module]):

        super().__init__(operations=operations)

    def forward(
        self, state: State, values: dict[str, Tensor] | ParameterDict = dict()
    ) -> State:
        """
        Apply the operations multiplying by the parameter values.

        Arguments:
            state: Input state.
            values: Parameter value.

        Returns:
            The transformed state.
        """
        return reduce(add, (op(state, values) for op in self.operations))

    def tensor(
        self, values: dict = {}, n_qubits: int | None = None, diagonal: bool = False
    ) -> Tensor:
        """
        Get the corresponding sum of unitaries over n_qubits.

        Arguments:
            values: Parameter value.
            n_qubits: The number of qubits the unitary is represented over.
            Can be higher than the number of qubit support.
            diagonal: Whether the operation is diagonal.


        Returns:
            The unitary representation.
        Raises:
            NotImplementedError for the diagonal case.
        """
        if n_qubits is None:
            n_qubits = max(self.qubit_support) + 1
        mat = torch.zeros((2, 2, 1), device=self.device)
        for _ in range(n_qubits - 1):
            mat = torch.kron(mat, torch.zeros((2, 2, 1), device=self.device))
        return reduce(
            add, (op.tensor(values, n_qubits, diagonal) for op in self.operations), mat
        )


class Observable(Sequence):
    """
    The Observable :math:`O` represents an operator from which
    we can extract expectation values from quantum states.

    Given an input state :math:`\\ket\\rangle`, the expectation value with :math:`O` is defined as
    :math:`\\langle\\bra|O\\ket\\rangle`

    Attributes:
        operations: List of operations.
        n_qubits: Number of qubits it is defined on.
    """

    def __init__(
        self,
        n_qubits: int | None,
        operations: list[Module] | Primitive | Sequence,
    ):
        super().__init__(operations)
        if n_qubits is None:
            n_qubits = max(self.qubit_support) + 1
        self.n_qubits = n_qubits

    def run(self, state: Tensor, values: dict[str, Tensor]) -> State:
        """
        Apply the observable onto a state to obtain :math:`\\|O\\ket\\rangle`.

        Arguments:
            state: Input state.
            values: Values of parameters.

        Returns:
            The transformed state.
        """
        for op in self.operations:
            state = op(state, values)
        return state

    def forward(
        self, state: Tensor, values: dict[str, Tensor] | ParameterDict = dict()
    ) -> Tensor:
        """Calculate the inner product :math:`\\langle\\bra|O\\ket\\rangle`

        Arguments:
            state: Input state.
            values: Values of parameters.

        Returns:
            The expectation value.
        """
        return inner_prod(state, self.run(state, values)).real


class DiagonalObservable(Primitive):
    """
    Special case of diagonal observables where computation is simpler.
    We simply do a element-wise vector-product instead of a tensordot.

    Attributes:
        pauli: The tensor representation from Primitive.
        qubit_support: Qubits the operator acts on.
        n_qubits: Number of qubits the operator is defined on.
    """

    def __init__(
        self,
        n_qubits: int | None,
        operations: list[Module] | Primitive | Sequence,
        to_sparse: bool = False,
    ):
        """Initializes the DiagonalObservable.

        Arguments:
            n_qubits: Number of qubits the operator is defined on.
            operations: Operations defining the observable.
            to_sparse: Whether to convert the operator to its sparse representation or not.
        """
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
        """
        Apply the observable onto a state to obtain :math:`\\|O\\ket\\rangle`.

        We flatten the state, do a element-wise multiplication with the diagonal hamiltonian
        and reshape it back to pyq-shape.


        Arguments:
            state: Input state.
            values: Values of parameters. Unused here.

        Returns:
            The transformed state.
        """
        return torch.einsum(
            "ij,ib->ib", self.pauli, state.flatten(start_dim=0, end_dim=-2)
        ).reshape([2] * self.n_qubits + [state.shape[-1]])

    def forward(
        self, state: Tensor, values: dict[str, Tensor] | ParameterDict = dict()
    ) -> Tensor:
        """Calculate the inner product :math:`\\langle\\bra|O\\ket\\rangle`

        Arguments:
            state: Input state.
            values: Values of parameters.

        Returns:
            The expectation value.
        """
        return inner_prod(state, self.run(state, values)).real


def is_diag_hamiltonian(hamiltonian: Operator, atol: Tensor = ATOL) -> bool:
    """
    Returns True if the batched tensors H are diagonal.

    Arguments:
        H: Input tensors.
        atol: Tolerance for near-zero values.

    Returns:
        True if diagonal, else False.
    """
    diag_check = torch.tensor(
        [
            is_diag(hamiltonian[..., i], atol)
            for i in range(hamiltonian.shape[BATCH_DIM])
        ],
        device=hamiltonian.device,
    )
    return bool(torch.prod(diag_check))


def evolve(hamiltonian: Operator, time_evolution: Tensor) -> Operator:
    """Get the evolved operator.

    For a hamiltonian :math:`H` and a time evolution :math:`t`, returns :math:`exp(-i H, t)`

    Arguments:
        hamiltonian: The operator :math:`H` for evolution.
        time_evolution: The evolution time :math:`t`.

    Returns:
        The evolution operator.
    """
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
    """
    The HamiltonianEvolution corresponds to :math:`t`, returns :math:`exp(-i H, t)` where
    a hamiltonian/generator :math:`H` and a time evolution :math:`t` are given.

    We can create such operation by passing different generator types:
        - A tensor representation of the generator,
        - A string when we consider the generator as a symbol.
        - Operations as a single primitive or a sequence, parameterized or not.

    Attributes:
        generator_type: This attribute informs on how the generator is inputed
        and sets the logic for applying hamiltonian evolution.
        time: The evolution time :math:`t`.
        operations: List of operations.
    """

    def __init__(
        self,
        generator: TGenerator,
        time: Tensor | str,
        qubit_support: Tuple[int, ...] | None = None,
        generator_parametric: bool = False,
    ):
        """Initializes the HamiltonianEvolution.
        Depending on the generator argument, set the type and set the right generator getter.

        Arguments:
            generator: The generator :math:`H`.
            time: The evolution time :math:`t`.
            qubit_support: The qubits the operator acts on.
            generator_parametric: Whether the generator is parametric or not.
        """
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
            qubit_support = (
                generator.qubit_support
                if (
                    not qubit_support
                    or len(qubit_support) <= len(generator.qubit_support)
                )
                else qubit_support
            )
            if generator_parametric:
                generator = [generator]
                self.generator_type = GeneratorType.PARAMETRIC_OPERATION
            else:
                generator = [
                    Primitive(
                        generator.tensor({}, len(qubit_support)),
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
        """Returns the operations making the generator.

        Returns:
            The generator as a ModuleList.
        """
        return self.operations

    def _symbolic_generator(self, values: dict) -> Operator:
        """Returns the generator for the SYMBOL case.

        Returns:
            The generator as a tensor.
        """
        hamiltonian = values[self.generator_symbol]
        # add batch dim
        if len(hamiltonian.shape) == 2:
            return hamiltonian.unsqueeze(2)
        # cases when the batchdim is at index 0 instead of 2
        if len(hamiltonian.shape) == 3 and (
            hamiltonian.shape[0] != hamiltonian.shape[1]
        ):
            return torch.transpose(hamiltonian, 0, 2)
        if len(hamiltonian.shape) == 4 and (
            hamiltonian.shape[0] != hamiltonian.shape[1]
        ):
            return torch.permute(hamiltonian.squeeze(3), (1, 2, 0))
        return hamiltonian

    def _tensor_generator(self, values: dict = {}) -> Operator:
        """Returns the generator for the TENSOR and OPERATION cases.

        Arguments:
            values: Non-used argument for consistency with other generator getters.

        Returns:
            The generator as a tensor.
        """
        return self.generator[0].pauli

    def _parametric_generator(self, values: dict) -> Operator:
        """Returns the generator for the PARAMETRIC_OPERATION case.

        Arguments:
            values: Values of parameters.

        Returns:
            The generator as a tensor.
        """
        return self.generator[0].tensor(values, len(self.qubit_support))

    @property
    def create_hamiltonian(self) -> Callable[[dict], Operator]:
        """A utility method for setting the right generator getter depending on the init case.

        Returns:
            The right generator getter.
        """
        return self._generator_map[self.generator_type]

    def forward(
        self, state: Tensor, values: dict[str, Tensor] | ParameterDict = dict()
    ) -> State:
        """
        Apply the hamiltonian evolution with input parameter values.

        Arguments:
            state: Input state.
            values: Values of parameters.

        Returns:
            The transformed state.
        """
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
    ) -> Operator:
        """Get the corresponding unitary over n_qubits.

        Arguments:
            values: Parameter value.
            n_qubits: The number of qubits the unitary is represented over.
            Can be higher than the number of qubit support.
            diagonal: Whether the operation is diagonal.


        Returns:
            The unitary representation.
        Raises:
            NotImplementedError for the diagonal case.
        """
        if diagonal:
            raise NotImplementedError
        if n_qubits is None:
            n_qubits = max(self.qubit_support) + 1
        hamiltonian: torch.Tensor = self.create_hamiltonian(values)
        time_evolution: torch.Tensor = (
            values[self.time] if isinstance(self.time, str) else self.time
        )
        return evolve(hamiltonian, time_evolution)
