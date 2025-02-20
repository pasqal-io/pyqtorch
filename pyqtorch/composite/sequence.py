from __future__ import annotations

import logging
from functools import reduce
from logging import getLogger
from operator import add
from typing import Any, Iterator

import torch
from numpy import int64
from torch import Tensor, complex128, einsum
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch.nn import Module, ModuleList, ParameterDict

from pyqtorch.embed import Embedding
from pyqtorch.matrices import _dagger, add_batch_dim
from pyqtorch.utils import (
    State,
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

        self.is_diagonal = all([op.is_diagonal for op in self.flatten()])

    @property
    def qubit_support(self) -> tuple:
        return tuple(sorted(self._qubit_support))

    def __iter__(self) -> Iterator:
        return iter(self.operations)

    def __len__(self) -> int:
        return len(self.operations)

    def __hash__(self) -> int:
        return hash(
            reduce(add, (hash(op) for op in self.operations))
            if len(self.operations) > 0
            else self.operations
        )

    def forward(
        self,
        state: Tensor,
        values: dict[str, Tensor] | ParameterDict | None = None,
        embedding: Embedding | None = None,
    ) -> State:
        values = values or dict()
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
        values: dict[str, Tensor] | None = None,
        embedding: Embedding | None = None,
        full_support: tuple[int, ...] | None = None,
        diagonal: bool = False,
    ) -> Tensor:
        if full_support is None:
            full_support = self.qubit_support
        elif not set(self.qubit_support).issubset(set(full_support)):
            raise ValueError(
                "Expanding tensor operation requires a `full_support` argument "
                "larger than or equal to the `qubit_support`."
            )
        use_diagonal = diagonal and self.is_diagonal
        values = values or dict()
        if not use_diagonal:
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
        else:
            mat = torch.ones(
                2 ** len(full_support), dtype=self.dtype, device=self.device
            ).unsqueeze(1)
            return reduce(
                lambda t0, t1: t0 * t1,
                (
                    op.tensor(values, embedding, full_support, diagonal=True)
                    for op in self.operations
                ),
                mat,
            )

    def dagger(
        self,
        values: dict[str, Tensor] | Tensor | None = None,
        embedding: Embedding | None = None,
        diagonal: bool = False,
    ) -> Tensor:
        values = values or dict()
        use_diagonal = diagonal and self.is_diagonal
        return _dagger(self.tensor(values, embedding, diagonal=use_diagonal))
