from __future__ import annotations

from math import log2
from string import ascii_letters as ABC

from numpy import array
from numpy.typing import NDArray
from torch import Tensor, einsum

from pyqtorch.matrices import _dagger
from pyqtorch.utils import DensityMatrix, permute_basis, permute_state

ABC_ARRAY: NDArray = array(list(ABC))


def apply_operator(
    state: Tensor,
    operator: Tensor,
    qubit_support: tuple[int, ...] | list[int],
    diagonal: bool = False,
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
        diagonal: Whether operator is diagonal or not.

    Returns:
        State after applying 'operator'.
    """
    qubit_support = list(qubit_support)
    n_qubits = len(state.size()) - 1
    n_support = len(qubit_support)
    n_state_dims = n_qubits + 1

    in_state_dims = ABC_ARRAY[0:n_state_dims].copy()
    if not diagonal:
        operator = operator.view([2] * n_support * 2 + [operator.size(-1)])
        operator_dims = ABC_ARRAY[
            n_state_dims : n_state_dims + 2 * n_support + 1
        ].copy()
        operator_dims[n_support : 2 * n_support] = in_state_dims[qubit_support]
    else:
        operator = operator.view([2] * n_support + [operator.size(-1)])
        operator_dims = ABC_ARRAY[n_state_dims : n_state_dims + n_support + 1].copy()
        operator_dims[:n_support] = in_state_dims[qubit_support]

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
    diagonal: bool = False,
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
        diagonal: Whether operator is diagonal or not.

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
    if not diagonal:
        einsum_expr = "ijb,jkb->ikb"
    else:
        einsum_expr = "jb,jkb->jkb"
    result = einsum(einsum_expr, operator, state).reshape([2] * n_qubits + [batch_size])
    return permute_state(result, support_perm, inv=True)


def apply_operator_dm(
    state: DensityMatrix,
    operator: Tensor,
    qubit_support: tuple[int, ...] | list[int],
    diagonal: bool = False,
) -> Tensor:
    """
    Apply an operator to a density matrix on a given qubit suport, i.e., compute:

    OP.DM.OP.dagger()

    Args:
        state: State to operate on.
        operator: Tensor to contract over 'state'.
        qubit_support: Tuple of qubits on which to apply the 'operator' to.
        diagonal: Whether operator is diagonal or not.

    Returns:
        DensityMatrix: The resulting density matrix after applying the operator.
    """

    if not isinstance(state, DensityMatrix):
        raise TypeError("Function apply_operator_dm requires a density matrix state.")

    n_qubits = int(log2(state.size()[0]))
    n_support = len(qubit_support)
    batch_size = max(state.size(-1), operator.size(-1))
    full_support = tuple(range(n_qubits))
    support_perm = tuple(sorted(qubit_support)) + tuple(
        set(full_support) - set(qubit_support)
    )
    state = permute_basis(state, support_perm)

    # There is probably a smart way to represent the lines below in a single einsum...
    if not diagonal:
        einsum_expr = "ijb,jkb->ikb"
    else:
        einsum_expr = "jb,jkb->jkb"
    state = state.reshape(
        [2**n_support, (2 ** (2 * n_qubits - n_support)), state.size(-1)]
    )
    state = einsum(einsum_expr, operator, state).reshape(
        [2**n_qubits, 2**n_qubits, batch_size]
    )
    state = _dagger(state).reshape(
        [2**n_support, (2 ** (2 * n_qubits - n_support)), state.size(-1)]
    )
    state = _dagger(
        einsum(einsum_expr, operator, state).reshape(
            [2**n_qubits, 2**n_qubits, batch_size]
        )
    )
    return permute_basis(state, support_perm, inv=True)


def operator_product(
    op1: Tensor,
    supp1: tuple[int, ...],
    op2: Tensor,
    supp2: tuple[int, ...],
) -> Tensor:
    """
    Operator product based on block matrix multiplication.

    E.g., for some small operator S and a big operator with 4 partitions A, B, C, D:

    |S 0|.|A B| = |S.A S.B|
    |0 S| |C D|   |S.C S.D|

    or

    |A B|.|S 0| = |A.S B.S|
    |C D|.|0 S|   |C.S D.S|

    The same generalizes for different sizes of the big operator. Note that the block
    diagonal matrix is never computed. Instead, the big operator is permuted and
    reshaped into a wide matrix:

    W = [A B C D]

    And then the result is computed as S.W, reshaped back into a square matrix, and
    permuted back into the original ordering.
    """

    if supp1 == supp2:
        return einsum("ijb,jkb->ikb", op1, op2)

    if len(supp1) < len(supp2):
        small_op, small_supp = op1, supp1
        big_op, big_supp = op2, supp2
        transpose = False
    else:
        small_op, small_supp = _dagger(op2), supp2
        big_op, big_supp = _dagger(op1), supp1
        transpose = True

    if not set(small_supp).issubset(set(big_supp)):
        raise ValueError("Operator product requires overlapping qubit supports.")

    n_big, n_small = len(big_supp), len(small_supp)
    batch_big, batch_small = big_op.size(-1), small_op.size(-1)
    batch_size = max(batch_big, batch_small)

    # Permute the large operator and reshape into a wide matrix
    support_perm = tuple(sorted(small_supp)) + tuple(set(big_supp) - set(small_supp))
    big_op = permute_basis(big_op, support_perm)
    big_op = big_op.reshape([2**n_small, (2 ** (2 * n_big - n_small)), batch_big])

    # Compute S.W and reshape back to square
    result = einsum("ijb,jkb->ikb", small_op, big_op).reshape(
        [2**n_big, 2**n_big, batch_size]
    )

    # Apply the inverse qubit permutation
    if transpose:
        return _dagger(permute_basis(result, support_perm, inv=True))
    else:
        return permute_basis(result, support_perm, inv=True)
