from __future__ import annotations

from functools import reduce
from logging import getLogger
from operator import add

from torch import Tensor
from torch.nn import Module, ModuleList, ParameterDict

from pyqtorch.utils import State

logger = getLogger(__name__)


class CompOp(ModuleList):
    def __init__(self, operations: list[Module]):
        if not isinstance(operations, list | ModuleList):
            raise TypeError("Please pass a list of individual operations.")
        super().__init__(operations)

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


class AddOp(CompOp):
    def __init__(self, operations: list[Module] | dict[str, Module]):
        self.param_map = {}
        self.inv_param_map = {}
        self.is_parameterized = False

        if isinstance(operations, dict):
            super().__init__(list(operations.values()))
            self.is_parameterized = True
            self.param_map = {i: k for i, k in enumerate(operations.keys())}
            self.inv_param_map = {k: i for i, k in self.param_map.items()}
        else:
            super().__init__(operations)

    def __getitem__(self, key: str | int) -> Module:
        index = self.inv_param_map[key] if isinstance(key, str) else key
        return super().__getitem__(index)

    def forward(self, state: State, values: dict[str, Tensor] | ParameterDict = {}) -> State:
        if self.is_parameterized:
            coeffs = [values[self.param_map[i]] for i in range(len(self))]
            return reduce(add, [coeffs[i] * op(state, values) for i, op in enumerate(self)])
        else:
            return reduce(add, [op(state, values) for op in self])
