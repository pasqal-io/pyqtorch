from __future__ import annotations

import pytest
import torch

from pyqtorch import RX, RY, ConcretizedCallable, Scale, Sequence, X
from pyqtorch.utils import heaviside, is_parametric


@pytest.mark.parametrize(
    "operation, result",
    [
        (RX(0, "x"), True),
        (RY(1, 0.5), False),
        (Scale(X(1), "y"), True),
        (Scale(X(1), 0.2), False),
        (Scale(X(1), ConcretizedCallable("mul", ["y", "x"])), True),
    ],
)
def test_is_parametric(operation: Sequence, result: bool) -> None:
    assert is_parametric(operation) == result


def test_heaviside() -> None:
    x = torch.linspace(-1, 1, 50)
    assert torch.allclose(heaviside(x), torch.heaviside(x, torch.tensor(0.0)))
