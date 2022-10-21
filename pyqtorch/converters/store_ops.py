from dataclasses import dataclass
from typing import List, Union


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
    name: str, targets: List[int], param: Union[float, List[float]] = None
) -> None:
    """Store an operation in the case saving its properties

    Args:
        name (str): _description_
        targets (List[int]): _description_
        param (Union[float, List[float]], optional): _description_. Defaults to None.
    """
    if param is not None:
        param = [float(p) for p in param] if type(param) == list else float(param)

    op = Operation(name=name, targets=targets, param=param)
    ops_cache.operations.append(op)
