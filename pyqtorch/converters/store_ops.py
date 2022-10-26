from dataclasses import dataclass
from typing import List, Union

import torch


@dataclass
class Operation:
    name: str
    targets: List[int]
    param: Union[List[float], float, None] = None


class OpsCache:

    # make it a singleton
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(OpsCache, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.operations: List[Operation] = []
        self.nqubits: int = 0
        self.enabled: bool = False

    def enable(self):
        if not self.enabled:
            self.enabled = True

    def clear(self, *args, **kwargs):
        """Clear the current circuit visualization"""
        self.operations = []
        self.nqubits = 0


ops_cache = OpsCache()


def store_operation(
    name: str, targets: List[int], param: Union[float, List[float], torch.Tensor] = None
) -> None:
    """Store an operation in the case saving its properties

    Args:
        name (str): _description_
        targets (List[int]): _description_
        param (Union[float, List[float]], optional): _description_. Defaults to None.
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
