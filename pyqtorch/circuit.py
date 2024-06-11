from __future__ import annotations

from logging import getLogger
from typing import Any, Iterator

from torch import Tensor, bernoulli, complex128, tensor
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch.nn import Module, ModuleList, ParameterDict

from pyqtorch.utils import DiffMode, DropoutMode, State, inner_prod, zero_state

logger = getLogger(__name__)


class DropoutQuantumCircuit(Module):
    def __init__(
        self,
        n_qubits: int,
        operations: list[Module],
        dropout_mode: DropoutMode = DropoutMode.NONE,
        dropout_prob: float = 1.0,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.operations = ModuleList(operations)
        self.dropout_mode = dropout_mode
        self.dropout_prob = dropout_prob
        self._device = torch_device("cpu")
        self._dtype = complex128
        if len(self.operations) > 0:
            try:
                self._device = next(iter(set((op.device for op in self.operations))))
            except StopIteration:
                pass

        self.dropout_fn = (
            None if self.dropout_mode == "none" else self.get_dropout_fn(self.dropout_mode)
        )

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

    def run(self, state: State = None, values: dict[str, Tensor] | ParameterDict = {}) -> State:
        if state is None:
            state = self.init_state()
        return self.forward(state, values)

    def forward(self, state: State, values: dict[str, Tensor] | ParameterDict = {}) -> State:
        if self.dropout_fn:
            state = self.dropout_fn(state, values)
        else:
            for op in self.operations:
                state = op(state, values)

        return state

    def get_dropout_fn(self, dropout_mode: str) -> Any:
        dropout_fn = getattr(self, dropout_mode)
        if not callable(dropout_fn):
            raise ValueError(f"No dropout mode named {dropout_mode} found")
        return dropout_fn

    def rotational_dropout(
        self, state: State = None, values: dict[str, Tensor] | ParameterDict = {}
    ) -> State:
        for op in self.operations:
            if hasattr(op, "param_name"):
                if values[op.param_name].requires_grad:
                    keep = 1 - bernoulli(tensor(self.dropout_prob))  # type: ignore
                    if keep:
                        state = op(state, values)

                else:
                    state = op(state, values)
            else:
                state = op(state, values)

        return state

    def entangling_dropout(
        self, state: State = None, values: dict[str, Tensor] | ParameterDict = {}
    ) -> State:
        for op in self.operations:
            if not (hasattr(op, "param_name")):
                keep = bool(1 - bernoulli(tensor(self.dropout_prob)))  # type: ignore
                if keep:
                    state = op(state, values)

                else:
                    state = op(state, values)
            else:
                state = op(state, values)

        return state

    def canonical_fwd_dropout(
        self, state: State = None, values: dict[str, Tensor] | ParameterDict = {}
    ) -> State:
        entanglers_to_drop = dict.fromkeys(range(state.ndim - 1), 0)  # type: ignore
        for op in self.operations:
            if hasattr(op, "param_name"):
                if values[op.param_name].requires_grad:
                    keep = bool(1 - bernoulli(tensor(self.dropout_prob)))  # type: ignore
                    if keep:
                        state = op(state, values)
                    else:
                        entanglers_to_drop[op.target] = 1
                else:
                    state = op(state, values)
            else:
                if entanglers_to_drop[op.control[0]] == 1:
                    entanglers_to_drop[op.control[0]] = 0
                else:
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

    def reverse(self) -> QuantumCircuit:
        return QuantumCircuit(self.n_qubits, ModuleList(list(reversed(self.operations))))

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
