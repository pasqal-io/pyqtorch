from __future__ import annotations

from typing import Callable
from .utils import StrEnum

class DecomposeMode(StrEnum):
    """
    Which Differentiation method to use.
    """

    NODECOMPOSE = ""
    """Do not use any grouping for quantum operations."""

decompose_mode_callable: dict[str, Callable] = {
    DecomposeMode.NODECOMPOSE: lambda op: [op]
}