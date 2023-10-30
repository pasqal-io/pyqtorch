from __future__ import annotations

from string import ascii_letters as ABC
from typing import Tuple

from numpy import array
from numpy.typing import NDArray
from torch import einsum

from pyqtorch.utils import Operator, State

ABC_ARRAY: NDArray = array(list(ABC))


def apply_operator(
    state: State,
    operator: Operator,
    qubits: Tuple[int, ...] | list[int],
    n_qubits: int = None,
    batch_size: int = None,
) -> State:
    qubits = list(qubits)
    if n_qubits is None:
        n_qubits = len(state.size()) - 1
    if batch_size is None:
        batch_size = state.size(-1)
    n_support = len(qubits)
    n_state_dims = n_qubits + 1
    operator = operator.view([2] * n_support * 2 + [operator.size(-1)])
    in_state_dims = ABC_ARRAY[0:n_state_dims].copy()
    operator_dims = ABC_ARRAY[n_state_dims : n_state_dims + 2 * n_support + 1].copy()
    operator_dims[n_support : 2 * n_support] = in_state_dims[qubits]
    operator_dims[-1] = in_state_dims[-1]
    out_state_dims = in_state_dims.copy()
    out_state_dims[qubits] = operator_dims[0:n_support]
    operator_dims, in_state_dims, out_state_dims = list(
        map(lambda e: "".join(list(e)), [operator_dims, in_state_dims, out_state_dims])
    )
    return einsum(f"{operator_dims},{in_state_dims}->{out_state_dims}", operator, state)
