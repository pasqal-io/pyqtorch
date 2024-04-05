from __future__ import annotations

from functools import reduce
from logging import getLogger
from operator import add
from typing import Iterator

from torch import Tensor, complex128
from torch import device as torch_device
from torch import dtype as torch_dtype
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
            self._key_map = {k: i for i, k in enumerate(operations.keys())}
            self._inv_key_map = {i: k for i, k in enumerate(operations.keys())}
            super().__init__(list(operations.values()))

    def __getitem__(self, key: str | int) -> Module:
        index = self._key_map[key] if isinstance(key, str) else key
        return super().__getitem__(index)


class CompositeOperation(Module):
    def __init__(self, operations: list[Module] | dict[str, Module]):
        super().__init__()
        self.operations = OpContainer(operations)
        self._device = torch_device("cpu")
        self._dtype = complex128
        if len(self.operations) > 0:
            try:
                self._device = next(iter(set((op.device for op in self.operations))))
            except StopIteration:
                pass

    @property
    def qubit_support(self) -> tuple[int, ...]:
        return tuple(set(sum([op.qubit_support for op in self.operations], ())))

    def __iter__(self) -> Iterator:
        return iter(self.operations)

    def __key(self) -> tuple:
        return self.qubit_support

    def __hash__(self) -> int:
        return hash(self.__key())

    @property
    def device(self) -> torch_device:
        return self._device

    @property
    def dtype(self) -> torch_dtype:
        return self._dtype

    # def to(self, *args: Any, **kwargs: Any) -> CompositeOperation:
    #    self.operations = ModuleList([op.to(*args, **kwargs) for op in self.operations])
    #    if len(self.operations) > 0:
    #        self._device = self.operations[0].device
    #        self._dtype = self.operations[0].dtype
    #    return self


class SeqOperation(CompositeOperation):
    def forward(self, state: State, values: dict[str, Tensor] | ParameterDict = {}) -> State:
        for op in self.operations:
            state = op(state, values)
        return state


class AddOperation(CompositeOperation):
    def forward(self, state: State, values: dict[str, Tensor] | ParameterDict = {}) -> State:
        if self.operations.is_parameterized:
            result_list = []
            for i, op in enumerate(self.operations):
                param_key = self.operations._inv_key_map[i]
                result_list.append(values[param_key] * op(state, values))
            return reduce(add, result_list)
        else:
            return reduce(add, [op(state, values) for op in self.operations])
