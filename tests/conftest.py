from __future__ import annotations

from typing import Any

import pytest

from pyqtorch.primitive import H, I, Primitive, S, T, X, Y, Z


@pytest.fixture(params=[I, X, Y, Z, H, T, S])
def gate(request: Primitive) -> Any:
    return request.param
