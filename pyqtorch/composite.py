from __future__ import annotations

from functools import reduce
from logging import getLogger
from operator import add
from typing import Iterator

from torch import Tensor
from torch.nn import Module, ModuleList, ParameterDict

from pyqtorch.utils import State

logger = getLogger(__name__)


class OpContainer(ModuleList):
    def __init__(self, operations: list[Module] | dict[str, Module]):
        if isinstance(operations, list):
            self.is_parameterized = False
            super().__init__(operations)
        if isinstance(operations, dict):
            self.is_parameterized = True
            self.key_map = {k: i for i, k in enumerate(operations.keys())}
            self.inv_key_map = {i: k for i, k in enumerate(operations.keys())}
            super().__init__(list(operations.values()))

    def __getitem__(self, key: str | int) -> Module:
        index = self.key_map[key] if isinstance(key, str) else key
        return super().__getitem__(index)

    def forward(self, state: State, values: dict[str, Tensor] | ParameterDict = {}) -> State:
        for op in self:
            state = op(state, values)
        return state


class CompositeOperation(Module):
    def __init__(self, operations: list[Module] | dict[str, Module]):
        super().__init__()
        self.operations = OpContainer(operations)

    @property
    def qubit_support(self) -> tuple[int, ...]:
        return tuple(set(sum([op.qubit_support for op in self.operations], ())))

    def __iter__(self) -> Iterator:
        return iter(self.operations)

    def __key(self) -> tuple:
        return self.qubit_support

    def __hash__(self) -> int:
        return hash(self.__key())


class ApplyOp(CompositeOperation):
    def forward(self, state: State, values: dict[str, Tensor] | ParameterDict = {}) -> State:
        for op in self.operations:
            state = op(state, values)
        return state


class AddOp(CompositeOperation):
    def forward(self, state: State, values: dict[str, Tensor] | ParameterDict = {}) -> State:
        if self.operations.is_parameterized:
            result_list = []
            for i, op in enumerate(self.operations):
                param_key = self.operations.inv_key_map[i]
                result_list.append(values[param_key] * op(state, values))
            return reduce(add, result_list)
        else:
            return reduce(add, [op(state, values) for op in self.operations])
