from __future__ import annotations

import logging
from collections import Counter
from functools import reduce
from logging import getLogger
from operator import add
from typing import Any, Generator, Iterator, NoReturn

import torch
from numpy import int64
from torch import Tensor, bernoulli, complex128, einsum, rand, tensor
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch.nn import Module, ModuleList, ParameterDict

from pyqtorch.apply import apply_operator
from pyqtorch.embed import Embedding
from pyqtorch.matrices import _dagger, add_batch_dim
from pyqtorch.parametric import RX, RY, Parametric
from pyqtorch.primitive import CNOT, Primitive
from pyqtorch.utils import (
    DensityMatrix,
    DropoutMode,
    State,
    product_state,
    sample_multinomial,
    zero_state,
)

logger = getLogger(__name__)


def forward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Forward complete")
    torch.cuda.nvtx.range_pop()


def pre_forward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Executing forward")
    torch.cuda.nvtx.range_push("QuantumCircuit.forward")


def backward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Backward complete")
    torch.cuda.nvtx.range_pop()


def pre_backward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Executed backward")
    torch.cuda.nvtx.range_push("QuantumCircuit.backward")


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
        logger.debug("QuantumCircuit initialized")
        if logger.isEnabledFor(logging.DEBUG):
            # When Debugging let's add logging and NVTX markers
            # WARNING: incurs performance penalty
            self.register_forward_hook(forward_hook, always_call=True)
            self.register_full_backward_hook(backward_hook)
            self.register_forward_pre_hook(pre_forward_hook)
            self.register_full_backward_pre_hook(pre_backward_hook)

        self._qubit_support = tuple(
            set(
                sum(
                    [op.qubit_support for op in self.operations],
                    (),
                )
            )
        )

        self._qubit_support = tuple(
            map(
                lambda x: x if isinstance(x, (int, int64)) else x[0],
                self._qubit_support,
            )
        )
        assert all(
            [isinstance(q, (int, int64)) for q in self._qubit_support]
        )  # TODO fix numpy.int issue

    @property
    def qubit_support(self) -> tuple:
        return self._qubit_support

    def __iter__(self) -> Iterator:
        return iter(self.operations)

    def __len__(self) -> int:
        return len(self.operations)

    def __hash__(self) -> int:
        return hash(reduce(add, (hash(op) for op in self.operations)))

    def forward(
        self,
        state: Tensor,
        values: dict[str, Tensor] | ParameterDict = dict(),
        embedding: Embedding | None = None,
    ) -> State:
        for op in self.operations:
            state = op(state, values, embedding)
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

    def flatten(self) -> ModuleList:
        ops = []
        for op in self.operations:
            if isinstance(op, Sequence):
                ops += op.flatten()
            else:
                ops.append(op)
        return ModuleList(ops)

    def tensor(
        self,
        values: dict[str, Tensor] = dict(),
        embedding: Embedding | None = None,
        full_support: tuple[int, ...] | None = None,
    ) -> Tensor:
        if full_support is None:
            full_support = self.qubit_support
        elif not set(self.qubit_support).issubset(set(full_support)):
            raise ValueError(
                "Expanding tensor operation requires a `full_support` argument "
                "larger than or equal to the `qubit_support`."
            )
        mat = torch.eye(
            2 ** len(full_support), dtype=self.dtype, device=self.device
        ).unsqueeze(2)
        return reduce(
            lambda t0, t1: einsum("ijb,jkb->ikb", t1, t0),
            (
                add_batch_dim(op.tensor(values, embedding, full_support))
                for op in self.operations
            ),
            mat,
        )

    def dagger(
        self,
        values: dict[str, Tensor] | Tensor = dict(),
        embedding: Embedding | None = None,
    ) -> Tensor:
        return _dagger(self.tensor(values, embedding))


class QuantumCircuit(Sequence):
    """A QuantumCircuit defining a register / number of qubits of the full system."""

    def __init__(self, n_qubits: int, operations: list[Module]):
        super().__init__(operations)
        self.n_qubits = n_qubits

    def run(
        self,
        state: Tensor = None,
        values: dict[str, Tensor] | ParameterDict = dict(),
        embedding: Embedding | None = None,
    ) -> State:
        if state is None:
            state = self.init_state()
        elif isinstance(state, str):
            state = self.state_from_bitstring(state)
        return self.forward(state, values, embedding)

    def __hash__(self) -> int:
        return hash(reduce(add, (hash(op) for op in self.operations))) + hash(
            self.n_qubits
        )

    def init_state(self, batch_size: int = 1) -> Tensor:
        return zero_state(
            self.n_qubits, batch_size, device=self.device, dtype=self.dtype
        )

    def state_from_bitstring(self, bitstring: str, batch_size: int = 1) -> Tensor:
        return product_state(bitstring, batch_size, self.device, self.dtype)

    def sample(
        self,
        state: Tensor = None,
        values: dict[str, Tensor] = dict(),
        n_shots: int = 1000,
        embedding: Embedding | None = None,
    ) -> list[Counter]:
        if n_shots < 1:
            raise ValueError(
                f"You can only call sample with a non-negative value for `n_shots`. Got {n_shots}."
            )

        with torch.no_grad():
            state = self.run(state=state, values=values, embedding=embedding)
            if isinstance(state, DensityMatrix):
                probs = torch.diagonal(state, dim1=0, dim2=1).real
            else:
                state = torch.flatten(
                    state,
                    start_dim=0,
                    end_dim=-2,
                ).t()
                probs = torch.abs(torch.pow(state, 2))

            probs = torch.pow(torch.abs(state), 2)
            return list(
                map(lambda p: sample_multinomial(p, self.n_qubits, n_shots), probs)
            )


class DropoutQuantumCircuit(QuantumCircuit):
    """Creates a quantum circuit able to perform quantum dropout, based on the work of https://arxiv.org/abs/2310.04120.
    Args:
        dropout_mode (DropoutMode): type of dropout to perform. Defaults to DropoutMode.ROTATIONAL
        dropout_prob (float): dropout probability. Defaults to 0.06.
    """

    def __init__(
        self,
        n_qubits: int,
        operations: list[Module],
        dropout_mode: DropoutMode = DropoutMode.ROTATIONAL,
        dropout_prob: float = 0.06,
    ):
        super().__init__(n_qubits, operations)
        self.dropout_mode = dropout_mode
        self.dropout_prob = dropout_prob

        self.dropout_fn = getattr(self, dropout_mode)

    def forward(
        self,
        state: State,
        values: dict[str, Tensor] | ParameterDict = dict(),
        embedding: Embedding | None = None,
    ) -> State:
        if self.training:
            state = self.dropout_fn(state, values)
        else:
            for op in self.operations:
                state = op(state, values)
        return state

    def rotational_dropout(
        self, state: State = None, values: dict[str, Tensor] | ParameterDict = dict()
    ) -> State:
        """Randomly drops entangling rotational gates.

        Args:
            state (State, optional): pure state vector . Defaults to None.
            values (dict[str, Tensor] | ParameterDict, optional): gate parameters. Defaults to {}.

        Returns:
            State: pure state vector
        """
        for op in self.operations:
            if not (
                (hasattr(op, "param_name"))
                and (values[op.param_name].requires_grad)
                and not (int(1 - bernoulli(tensor(self.dropout_prob))))
            ):
                state = op(state, values)

        return state

    def entangling_dropout(
        self, state: State = None, values: dict[str, Tensor] | ParameterDict = dict()
    ) -> State:
        """Randomly drops entangling gates.

        Args:
            state (State, optional): pure state vector. Defaults to None.
            values (dict[str, Tensor] | ParameterDict, optional): gate parameters. Defaults to {}.

        Returns:
            State: pure state vector
        """
        for op in self.operations:
            has_param = hasattr(op, "param_name")
            keep = int(1 - bernoulli(tensor(self.dropout_prob)))

            if has_param or keep:
                state = op(state, values)

        return state

    def canonical_fwd_dropout(
        self, state: State = None, values: dict[str, Tensor] | ParameterDict = dict()
    ) -> State:
        """Randomly drops rotational gates and next immediate entangling
        gates whose target bit is located on dropped rotational gates.

        Args:
            state (State, optional): pure state vector. Defaults to None.
            values (dict[str, Tensor] | ParameterDict, optional): gate parameters. Defaults to {}.

        Returns:
            State: pure state vector
        """
        entanglers_to_drop = dict.fromkeys(range(state.ndim - 1), 0)  # type: ignore
        for op in self.operations:
            if (
                hasattr(op, "param_name")
                and (values[op.param_name].requires_grad)
                and not (int(1 - bernoulli(tensor(self.dropout_prob))))
            ):
                entanglers_to_drop[op.target] = 1
            else:
                if not hasattr(op, "param_name") and (
                    entanglers_to_drop[op.control[0]] == 1
                ):
                    entanglers_to_drop[op.control[0]] = 0
                else:
                    state = op(state, values)

        return state


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
            raise TypeError(
                f"Require all operations to act on a single qubit. Got: {operations}."
            )

    def forward(
        self,
        state: Tensor,
        values: dict[str, Tensor] = dict(),
        embedding: Embedding | None = None,
    ) -> Tensor:
        batch_size = state.shape[-1]
        if values:
            batch_size = max(
                batch_size,
                max(
                    list(
                        map(
                            lambda t: len(t) if isinstance(t, Tensor) else 1,
                            values.values(),
                        )
                    )
                ),
            )
        return apply_operator(
            state,
            add_batch_dim(self.tensor(values, embedding), batch_size),
            self.qubits,
        )

    def tensor(
        self,
        values: dict[str, Tensor] = dict(),
        embedding: Embedding | None = None,
        full_support: tuple[int, ...] | None = None,
    ) -> Tensor:
        # We reverse the list of tensors here since matmul is not commutative.
        return reduce(
            lambda u0, u1: einsum("ijb,jkb->ikb", u0, u1),
            (
                op.tensor(values, embedding, full_support)
                for op in reversed(self.operations)
            ),
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
            layer += [
                Merge([fn(i, f"{param_name}_{next(idx)}") for fn in [RX, RY, RX]])
            ]
        ops += layer
        ops += [
            Sequence([CNOT(i % n_qubits, (i + 1) % n_qubits) for i in range(n_qubits)])
        ]
    params = ParameterDict(
        {f"{param_name}_{n}": rand(1, requires_grad=True) for n in range(next(idx))}
    )
    return ops, params
