from __future__ import annotations

from string import ascii_letters as ABC

from numpy import array
from numpy.typing import NDArray
from torch import Tensor, einsum

from pyqtorch.matrices import _dagger
from pyqtorch.utils import DensityMatrix, permute_state

ABC_ARRAY: NDArray = array(list(ABC))


def apply_operator(
    state: Tensor,
    operator: Tensor,
    qubit_support: tuple[int, ...] | list[int],
) -> Tensor:
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
        qubit_support: Tuple of qubits on which to apply the 'operator' to.

    Returns:
        State after applying 'operator'.
    """
    qubit_support = list(qubit_support)
    n_qubits = len(state.size()) - 1
    n_support = len(qubit_support)
    n_state_dims = n_qubits + 1
    operator = operator.view([2] * n_support * 2 + [operator.size(-1)])
    in_state_dims = ABC_ARRAY[0:n_state_dims].copy()
    operator_dims = ABC_ARRAY[n_state_dims : n_state_dims + 2 * n_support + 1].copy()
    operator_dims[n_support : 2 * n_support] = in_state_dims[qubit_support]
    operator_dims[-1] = in_state_dims[-1]
    out_state_dims = in_state_dims.copy()
    out_state_dims[qubit_support] = operator_dims[0:n_support]
    operator_dims, in_state_dims, out_state_dims = list(
        map(lambda e: "".join(list(e)), [operator_dims, in_state_dims, out_state_dims])
    )
    return einsum(f"{operator_dims},{in_state_dims}->{out_state_dims}", operator, state)


def apply_operator_permute(
    state: Tensor,
    operator: Tensor,
    qubit_support: tuple[int, ...] | list[int],
) -> Tensor:
    """NOTE: Currently not being used.

       Alternative apply operator function with a logic based on state permutations.
       Seems to be as fast as the current `apply_operator`. To be saved for now, we
       may want to switch to this one in the future if we wish to remove the state
       [2] * n_qubits shape and make the batch dimension the first one as the typical
       torch convention.

    Arguments:
        state: State to operate on.
        operator: Tensor to contract over 'state'.
        qubit_support: Tuple of qubits on which to apply the 'operator' to.

    Returns:
        State after applying 'operator'.
    """
    n_qubits = len(state.size()) - 1
    n_support = len(qubit_support)
    batch_size = max(state.size(-1), operator.size(-1))
    full_support = tuple(range(n_qubits))
    support_perm = list(sorted(qubit_support)) + list(
        set(full_support) - set(qubit_support)
    )
    state = permute_state(state, support_perm)
    state = state.reshape([2**n_support, 2 ** (n_qubits - n_support), state.size(-1)])
    result = einsum("ijb,jkb->ikb", operator, state).reshape(
        [2] * n_qubits + [batch_size]
    )
    return permute_state(result, support_perm, inv=True)


def apply_density_mat(op: Tensor, density_matrix: DensityMatrix) -> DensityMatrix:
    """
    Apply an operator to a density matrix, i.e., compute:
    op1 * density_matrix * op1_dagger.

    Args:
        op (Tensor): The operator to apply.
        density_matrix (DensityMatrix): The density matrix.

    Returns:
        DensityMatrix: The resulting density matrix after applying the operator and its dagger.
    """
    batch_size_op = op.size(-1)
    batch_size_dm = density_matrix.size(-1)
    if batch_size_dm > batch_size_op:
        # The other condition is impossible because
        # operators are always initialized with batch_size = 1.
        op = op.repeat(1, 1, batch_size_dm)
    return einsum("ijb,jkb,klb->ilb", op, density_matrix, _dagger(op))


def operator_product(op1: Tensor, op2: Tensor) -> Tensor:
    """
    Compute the product of two operators.
    `torch.bmm` is not suitable for our purposes because,
    in our convention, the batch_size is in the last dimension.

    Args:
        op1 (Tensor): The first operator.
        op2 (Tensor): The second operator.
    Returns:
        Tensor: The product of the two operators.
    """
    # ? Should we continue to adjust the batch here?
    # ? as now all gates are init with batch_size=1.
    batch_size_1 = op1.size(-1)
    batch_size_2 = op2.size(-1)
    if batch_size_1 > batch_size_2:
        op2 = op2.repeat(1, 1, batch_size_1)[:, :, :batch_size_1]
    elif batch_size_2 > batch_size_1:
        op1 = op1.repeat(1, 1, batch_size_2)[:, :, :batch_size_2]
    return einsum("ijb,jkb->ikb", op1, op2)
