from __future__ import annotations

from functools import reduce
from logging import getLogger
from operator import add

from torch import Tensor
from torch.nn import Module, ModuleList, ParameterDict

from pyqtorch.utils import State

logger = getLogger(__name__)


class SeqOps(ModuleList):
    def __init__(self, operations: list[Module] | dict[str, Module]):
        self.key_map = {}
        self.inv_key_map = {}

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

    @property
    def qubit_support(self) -> tuple[int, ...]:
        return tuple(set(sum([op.qubit_support for op in self], ())))

    def __key(self) -> tuple:
        return self.qubit_support

    def __hash__(self) -> int:
        return hash(self.__key())

    def forward(self, state: State, values: dict[str, Tensor] | ParameterDict = {}) -> State:
        for op in self:
            state = op(state, values)
        return state


class AddOps(SeqOps):
    def forward(self, state: State, values: dict[str, Tensor] | ParameterDict = {}) -> State:
        if self.is_parameterized:
            result_list = []
            for i, op in enumerate(self):
                param_key = self.inv_key_map[i]
                result_list.append(values[param_key] * op(state, values))
            return reduce(add, result_list)
        else:
            return reduce(add, [op(state, values) for op in self])
