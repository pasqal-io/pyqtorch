from __future__ import annotations

from string import ascii_letters as ABC

from numpy import array
from numpy.typing import NDArray
from torch import einsum

from pyqtorch.utils import Operator, State

ABC_ARRAY: NDArray = array(list(ABC))


def apply_operator(
    state: State,
    operator: Operator,
    qubits: list[int],
    n_qubits: int = None,
    batch_size: int = None,
) -> State:
    if n_qubits is None:
        n_qubits = len(state.size()) - 1
    if batch_size is None:
        batch_size = state.size(-1)
    n_support = len(qubits)
    n_state_dims = n_qubits + 1  # We have an additional batch dim.
    # View the operator tensor in the pyq shape, where each qubit represents one dimension.
    operator = operator.view([2] * n_support * 2 + [operator.size(-1)])
    # Assign letters to each dimension of the input state,
    #  i.e., dim 0: 'a', dim 1: 'b' etc.
    # Recall, the last dimension denotes the batch_dim.
    # Hence the n_state_dims'th letter stands for the batch dim.
    in_state_dims = ABC_ARRAY[0:n_state_dims].copy()
    # Assign letters to each dimension operator, note that we cant reuse the letters
    # we used for the state, hence we start indexing from n_state_dims.
    # Note that we need n_support + 1 letters for the operator since it also has a batch dim.
    op_dims = ABC_ARRAY[n_state_dims : n_state_dims + 2 * n_support + 1].copy()
    # Name the dimensions in the operator tensor the same as the
    # target qubit(s) we want to contract over in the input tensor.
    op_dims[n_support : 2 * n_support] = in_state_dims[qubits]
    # Both operator and state have batch dimensions so we give them the same name.
    op_dims[-1] = in_state_dims[-1]
    # Finally, the dimensions for the new state after the operator application are created.
    # Example: state= pyq.zero_state(2) will have shape : [2,2,1]
    # Its dimension in the einsum expression will receive the symbols: "abc" (c for the batch_dim)
    # If we apply: cnot= pyq.CNOT(0,1), its tensor will have shape: [2, 2, 2, 2, 1] after viewing.
    # We get 'deabc' for the operator and 'dec' for the output state.
    # resulting in 'deabc,abc->dec', which results in a contraction of
    # the operator dim 0 (denoted as d) over dim 0 of the state (denoted as a)
    # and contraction of dim 1 of the operator (e) over dim 1 of the state (b).
    out_state_dims = in_state_dims.copy()
    out_state_dims[qubits] = op_dims[0:n_support]
    op_dims, in_state_dims, out_state_dims = list(
        map(lambda e: "".join(list(e)), [op_dims, in_state_dims, out_state_dims])
    )
    return einsum(f"{op_dims},{in_state_dims}->{out_state_dims}", operator, state)
