from __future__ import annotations

from math import log2
from string import ascii_letters as ABC
from typing import Tuple

import torch
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


def apply_ope_ope(operator_1: torch.Tensor, operator_2: torch.Tensor) -> torch.Tensor:
    """
    Compute the product of two operators.
    The operators must have the same dimensions.

    Args:
        operator_1 (torch.Tensor): The first operator.
        operator_2 (torch.Tensor): The second operator.

    Returns:
        torch.Tensor: The product of the two operators.

    Raises:
        ValueError: If the number of qubits or the batch size differs between the two operators.
    """

    # Dimension verifications:
    n_qubits_1 = int(log2(operator_1.size(1)))
    n_qubits_2 = int(log2(operator_2.size(1)))
    batch_size_1 = operator_1.size(-1)
    batch_size_2 = operator_2.size(-1)
    if n_qubits_1 != n_qubits_2:
        raise ValueError("The number of qubit is different between the two operators.")
    if batch_size_1 != batch_size_2:
        raise ValueError("The number of batch is different between the two operators.")

    # Permute the batch size on first dimension to allow torch.bmm():
    def batch_first(operator: torch.Tensor) -> torch.Tensor:
        """
        Permute the operator's batch dimension on first dimension.

        Args:
        operator (torch.Tensor): Operator in size [2**n_qubits, 2**n_qubits,batch_size].

        Returns:
        torch.Tensor: Operator in size [batch_size, 2**n_qubits, 2**n_qubits].
        """
        batch_first_perm = (2, 0, 1)
        return torch.permute(operator, batch_first_perm)

    # Undo the permute since PyQ expects tensor.Size([2**n_qubits, 2**n_qubits,batch_size]):
    def batch_last(operator: torch.Tensor) -> torch.Tensor:
        """
        Permute the operator's batch dimension on last dimension.

        Args:
        operator (torch.Tensor): Operator in size [batch_size,2**n_qubits, 2**n_qubits].

        Returns:
        torch.Tensor: Operator in size [2**n_qubits, 2**n_qubits,batch_size].
        """
        undo_perm = (1, 2, 0)
        return torch.permute(operator, undo_perm)

    return batch_last(torch.bmm(batch_first(operator_1), batch_first(operator_2)))
