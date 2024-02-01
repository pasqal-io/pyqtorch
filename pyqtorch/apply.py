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
    """Applies an operator, i.e. a single tensor of shape [2, 2, ...], on a given state
       of shape [2 for _ in range(n_qubits)] for a given set of (target and control) qubits.

       Since dimension 'i' in 'state' corresponds to all amplitudes where qubit 'i' is 1,
       target and control qubits represent the dimensions over which to contract the 'operator'.
       Contraction means applying the 'dot' operation between the operator array and dimension 'i'
       of 'state, resulting in a new state where the result of the 'dot' operation has been moved to
       dimension 'i' of 'state'. To restore the former order of dimensions, the affected dimensions
       are moved to their original positions and the state is returned.

    Arguments:
        state: State to operate on.
        operator: Tensor to contract over 'state'.
        qubits: Tuple of qubits on which to apply the 'operator' to.
        n_qubits: The number of qubits of the full system.
        batch_size: Batch size of either state and or operators.

    Returns:
        State after applying 'operator'.
    """
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
