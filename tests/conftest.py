from __future__ import annotations

from typing import Any

import pytest
import torch

from pyqtorch.apply import apply_operator
from pyqtorch.primitive import H, I, Primitive, S, T, X, Y, Z


def _calc_mat_vec_wavefunction(
    block: Primitive, n_qubits: int, init_state: torch.Tensor, values: dict = {}
) -> torch.Tensor:
    """Get the result of applying the matrix representation of a block to an initial state.

    Args:
        block: The black operator to apply.
        n_qubits: Number of qubits in the circuit.
        init_state: Initial state to apply block on.
        values: Values of parameter if block is parametric.

    Returns:
        Tensor: The new wavefunction after applying the block.

    """
    mat = block.tensor(n_qubits=n_qubits, values=values)
    return apply_operator(
        init_state, mat, qubits=tuple(range(n_qubits)), n_qubits=n_qubits
    )


@pytest.fixture(params=[I, X, Y, Z, H, T, S])
def gate(request: Primitive) -> Any:
    return request.param
