from __future__ import annotations

from functools import reduce
from logging import getLogger
from operator import add
from typing import Any, Generator, Iterator, NoReturn

from torch import Tensor, complex128, einsum, rand
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch.nn import Module, ModuleList, ParameterDict

from pyqtorch.apply import apply_operator
from pyqtorch.parametric import RX, RY, Parametric
from pyqtorch.primitive import CNOT, Primitive
from pyqtorch.utils import State, add_batch_dim, zero_state

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

    def to(self, *args: Any, **kwargs: Any) -> Sequence:
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

    def flatten(self) -> ModuleList:
        ops = []
        for op in self.operations:
            if isinstance(op, Sequence):
                ops += op.operations
            else:
                ops.append(op)
        return ModuleList(ops)


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

        """

        if (
            isinstance(operations, (list, ModuleList))
            and all([isinstance(op, (Primitive, Parametric)) for op in operations])
            and all(list([len(op.qubit_support) == 1 for op in operations]))
            and len(list(set([op.qubit_support[0] for op in operations]))) == 1
        ):
            # We want all operations to act on the same qubit

            super().__init__(operations)
            self.qubits = operations[0].qubit_support
        else:
            raise TypeError(f"Require all operations to act on a single qubit. Got: {operations}.")

    def forward(self, state: Tensor, values: dict[str, Tensor] | None = None) -> Tensor:
        batch_size = state.shape[-1]
        if values:
            batch_size = max(batch_size, max(list(map(len, values.values()))))
        return apply_operator(
            state,
            self.unitary(values, batch_size),
            self.qubits,
        )

    def unitary(self, values: dict[str, Tensor] | None, batch_size: int) -> Tensor:
        # We reverse the list of tensors here since matmul is not commutative.
        return reduce(
            lambda u0, u1: einsum("ijb,jkb->ikb", u0, u1),
            (add_batch_dim(op.unitary(values), batch_size) for op in reversed(self.operations)),
        )


def hea(n_qubits: int, depth: int, param_name: str) -> tuple[ModuleList, ParameterDict]:
    def _idx() -> Generator[int, Any, NoReturn]:
        i = 0
        while True:
            yield i
            i += 1

    def idxer() -> Generator[int, Any, None]:
        yield from _idx()

    idx = idxer()
    ops = []
    for _ in range(depth):
        layer = []
        for i in range(n_qubits):
            layer += [Merge([fn(i, f"{param_name}_{next(idx)}") for fn in [RX, RY, RX]])]
        ops += layer
        ops += [Sequence([CNOT(i % n_qubits, (i + 1) % n_qubits) for i in range(n_qubits)])]
    params = ParameterDict(
        {f"{param_name}_{n}": rand(1, requires_grad=True) for n in range(next(idx))}
    )
    return ops, params
