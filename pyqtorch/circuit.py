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


class Sequence(Module):
    """A generic container for pyqtorch operations"""

    def __init__(self, operations: list[Module]):
        super().__init__()
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

    def __hash__(self) -> int:
        return hash(reduce(add, (hash(op) for op in self.operations)))

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

    def to(self, *args: Any, **kwargs: Any) -> QuantumCircuit:
        self.operations = ModuleList([op.to(*args, **kwargs) for op in self.operations])
        if len(self.operations) > 0:
            self._device = self.operations[0].device
            self._dtype = self.operations[0].dtype
        return self


class QuantumCircuit(Sequence):
    """A QuantumCircuit defining a register / number of qubits of the full system."""

    def __init__(self, n_qubits: int, operations: list[Module]):
        super().__init__(operations)
        self.n_qubits = n_qubits

    def run(self, state: State = None, values: dict[str, Tensor] | ParameterDict = {}) -> State:
        if state is None:
            state = self.init_state()
        return self.forward(state, values)

    def __hash__(self) -> int:
        return hash(reduce(add, (hash(op) for op in self.operations))) + hash(self.n_qubits)

    def init_state(self, batch_size: int = 1) -> Tensor:
        return zero_state(self.n_qubits, batch_size, device=self.device, dtype=self.dtype)


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


class Add(Sequence):
    """The 'add' operation applies all 'operations' to 'state' and returns the sum of states."""

    def __init__(self, operations: list[Module]):
        super().__init__(operations=operations)

    def forward(self, state: State, values: dict[str, Tensor] | ParameterDict = dict()) -> State:
        return reduce(add, (op(state, values) for op in self.operations))


class Merge(Sequence):
    def __init__(
        self,
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

            super().__init__(operations)
            self.qubits = operations[0].qubit_support
        else:
            raise TypeError(f"Require all operations to act on a single qubit. Got: {operations}.")

    def forward(self, state: Tensor, values: dict[str, Tensor] | None = None) -> Tensor:
        batch_size = state.shape[-1]
        if values and len(values) > 0:
            batch_size = max(batch_size, max(list(map(len, values.values()))))
        return apply_operator(
            state,
            self.unitary(values, batch_size),
            self.qubits,
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
