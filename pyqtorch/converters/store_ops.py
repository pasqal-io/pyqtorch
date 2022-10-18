from dataclasses import dataclass
from functools import wraps
import inspect
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

    def clear(self, *args, **kwargs):
        """Clear the current circuit visualization"""
        self.operations = []
        self.nqubits = 0


ops_cache = OpsCache()


def storable(func):
    """Decorator to make a circuit operation visualizable"""

    @wraps(func)
    def _wrapped(*args, **kwargs):

        try:
            ind_param = inspect.getfullargspec(func).args.index("theta")
            param = round(float(args[ind_param]), 4)
        except ValueError:
            param = None

        try:
            ind_target = inspect.getfullargspec(func).args.index("qubits")
            targets = args[ind_target]
        except ValueError:
            raise ValueError("The gate visualized should have a target!")

        op = Operation(name=func.__name__.lower(), targets=targets, param=param)
        ops_cache.operations.append(op)

        return func(*args, **kwargs)

    return _wrapped
