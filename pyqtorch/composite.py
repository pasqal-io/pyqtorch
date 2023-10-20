from __future__ import annotations

from functools import reduce
from operator import add
from typing import List

import torch

from pyqtorch.modules.operator import Operator


class Composite(torch.nn.Module):
    def __init__(self, operators: List[Operator]):
        self.operations = operators

    def forward(self, state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
        for op in self.operations:
            state = op(state, values)
        return state


class Add(Composite):
    def __init__(self, operators: List[Operator]):
        super().__init__(operators)

    def forward(self, state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return reduce(add, (op(state, values) for op in self.operations))
