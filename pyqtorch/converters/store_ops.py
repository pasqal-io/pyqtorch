from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Union

import torch
from numpy.typing import ArrayLike


@dataclass
class Operation:
    name: str
    targets: ArrayLike
    param: Union[List[float], float, torch.Tensor, None] = None


class OpsCache:
    # make it a singleton
    def __new__(cls) -> "OpsCache":
        if not hasattr(cls, "instance"):
            cls.instance = super(OpsCache, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        self.operations: List[Operation] = []
        self.nqubits: int = 0
        self.enabled: bool = False

    def enable(self) -> None:
        if not self.enabled:
            self.enabled = True

    def clear(self, *args: Any, **kwargs: Any) -> None:
        """Clear the current circuit visualization"""
        self.operations = []
        self.nqubits = 0


ops_cache = OpsCache()


def store_operation(
    name: str,
    targets: ArrayLike,
    param: Union[float, List[float], torch.Tensor, None] = None,
) -> None:
    """Store an operation in the case saving its properties

    Args:
        name (str): The name of the operation to store
        targets (ArrayLike): The target qubits
        param (Union[float, List[float]], optional): Optional parameter value
            for parametric operations
    """

    reshaped_par = param
    if param is not None:
        # taken into account the case of batched gates
        if isinstance(param, torch.Tensor):
            # FIXME: Taking only the first element for batched gates here
            reshaped_par = [float(param.reshape(-1)[0])]
        elif isinstance(param, List):
            reshaped_par = [float(p) for p in param]
        else:
            reshaped_par = [float(param)]

    op = Operation(name=name, targets=targets, param=reshaped_par)
    ops_cache.operations.append(op)
