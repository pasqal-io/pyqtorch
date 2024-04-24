import pytest
from typing import Any 

from pyqtorch.primitive import I, X, Y, Z, H, T, S, Primitive

@pytest.fixture(params=[I, X, Y, Z, H, T, S])
def gate(request: Primitive) -> Any:
    return request.param