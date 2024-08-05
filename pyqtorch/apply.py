from __future__ import annotations

from string import ascii_letters as ABC

from numpy import array
from numpy.typing import NDArray
from torch import Tensor, einsum

from pyqtorch.matrices import _dagger
from pyqtorch.utils import DensityMatrix

ABC_ARRAY: NDArray = array(list(ABC))


def apply_operator(
    state: Tensor,
    operator: Tensor,
    qubits: tuple[int, ...] | list[int],
    n_qubits: int | None = None,
    batch_size: int | None = None,
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
