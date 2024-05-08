from __future__ import annotations

from functools import reduce
from logging import getLogger
from operator import add
from typing import Any, Iterator

from torch import Tensor, bmm, complex128, ones_like
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch.nn import Module, ModuleList, ParameterDict

from pyqtorch.apply import apply_operator
from pyqtorch.matrices import _dagger
from pyqtorch.parametric import Parametric
from pyqtorch.primitive import Primitive
from pyqtorch.utils import DiffMode, State, batch_first, batch_last, inner_prod, zero_state

logger = getLogger(__name__)


class QuantumCircuit(Module):
    def __init__(self, n_qubits: int, operations: list[Module]):
        super().__init__()
        self.n_qubits = n_qubits
        self.operations = ModuleList(operations)
        self._device = torch_device("cpu")
        self._dtype = complex128
        if len(self.operations) > 0:
            try:
                self._device = next(iter(set((op.device for op in self.operations))))
            except StopIteration:
                pass

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

    def run(self, state: State = None, values: dict[str, Tensor] | ParameterDict = {}) -> State:
        if state is None:
            state = self.init_state()
        return self.forward(state, values)

    def forward(self, state: State, values: dict[str, Tensor] | ParameterDict = {}) -> State:
        for op in self.operations:
            state = op(state, values)
        return state

    @property
    def device(self) -> torch_device:
        return self._device

    @property
    def dtype(self) -> torch_dtype:
        return self._dtype

    def init_state(self, batch_size: int = 1) -> Tensor:
        return zero_state(self.n_qubits, batch_size, device=self.device, dtype=self.dtype)

    def to(self, *args: Any, **kwargs: Any) -> QuantumCircuit:
        self.operations = ModuleList([op.to(*args, **kwargs) for op in self.operations])
        if len(self.operations) > 0:
            self._device = self.operations[0].device
            self._dtype = self.operations[0].dtype
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


class Add(QuantumCircuit):
    def __init__(self, n_qubits: int, operations: list[Module]):
        """The 'add' operation applies all 'operations' to 'state' and returns the sum of states."""
        super().__init__(n_qubits=n_qubits, operations=operations)

    def forward(self, state: State, values: dict[str, Tensor] | ParameterDict = dict()) -> State:
        return reduce(add, (op(state, values) for op in self.operations))


class Merge(QuantumCircuit):
    def __init__(
        self,
        n_qubits: int,
        operations: list[Module],
    ):
        """
        Merge a sequence of single qubit operations acting on the same qubit into a single
        einsum operation.

        Arguments:
            operations: A list of single qubit operations.
            qubits: The target qubit.
            n_qubits: The number of qubits in the full system.

        """

        if (
            isinstance(operations, (list, ModuleList))
            and all([isinstance(op, (Primitive, Parametric)) for op in operations])
            and len(list(set([op.qubit_support[0] for op in operations]))) == 1
        ):
            # We want all operations to act on the same qubit

            super().__init__(n_qubits, operations)
            self.qubits = operations[0].qubit_support
        else:
            raise TypeError(f"Require all operations to act on a single qubit. Got: {operations}.")

    def forward(self, state: Tensor, values: dict[str, Tensor] | None = None) -> Tensor:
        batch_size = state.shape[-1]
        return apply_operator(
            state, self.unitary(values, batch_size), self.qubits, self.n_qubits, batch_size
        )

    def unitary(self, values: dict[str, Tensor] | None, batch_size: int) -> Tensor:
        def expand(operator: Tensor) -> Tensor:
            """In case we have a sequence of batched parametric gates mixed with primitive gates,
            we adjust the batch_dim of the primitive gates to match."""
            return (
                operator.repeat(1, 1, batch_size)
                if operator.shape != (2, 2, batch_size)
                else operator
            )

        # We reverse the list of tensors here since matmul is not commutative.
        return batch_last(
            reduce(
                bmm,
                (batch_first(expand(op.unitary(values))) for op in reversed(self.operations)),
            )
        )


class Scale(QuantumCircuit):
    def __init__(self, n_qubits: int, param_name: str, operation: Primitive):
        super().__init__(n_qubits, [operation])
        self.param_name = param_name
        self.qubit_support = operation.qubit_support

    def forward(self, state: Tensor, values: dict[str, Tensor] | ParameterDict = dict()) -> Tensor:
        return apply_operator(state, self.unitary(values), self.qubit_support, self.n_qubits)

    def unitary(self, values: dict[str, Tensor]) -> Tensor:
        thetas = values[self.param_name]
        return thetas * self.operations[0].unitary(values)

    def dagger(self, values: dict[str, Tensor]) -> Tensor:
        return _dagger(self.unitary(values))

    def jacobian(self, values: dict[str, Tensor]) -> Tensor:
        return values[self.param_name] * ones_like(self.unitary(values))
