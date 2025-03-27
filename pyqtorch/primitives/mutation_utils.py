from __future__ import annotations

import torch
from torch import Tensor


def mutate_separate_target(
    state: Tensor, target_qubit: int
) -> tuple[list[int], Tensor]:
    """Create a tensor separating the target components
    for a single-qubit gate for mutating an input state-vector.

    Args:
        state (Tensor): Input state.
        target_qubit (int): Target index.

    Returns:
        tuple[int list[int], Tensor]: The permutation indices with an intermediate state
            with separated target qubit.
    """
    n_qubits = len(state.shape) - 1
    perm = list(range(n_qubits + 1))
    perm[0], perm[target_qubit] = perm[target_qubit], perm[0]

    # Transpose the state
    state = state.permute(perm)

    # Reshape to separate the target qubit
    state = state.reshape(2, -1)
    return perm, state


def mutate_revert_modified(
    state: Tensor, original_shape: tuple[int], perm: list[int]
) -> Tensor:
    """After mutating a state given a single qubit gate, we revert back the new state
    to correspond to the `original_shape`.

    Args:
        state (Tensor): modified state by operation.
        original_shape (tuple[int]): original shape for reshapping.
        perm (list[int]): Permutation indices.

    Returns:
        Tensor: Mutated state.
    """
    # Reshape back to original structure
    state = state.reshape(original_shape)

    # Transpose back to original order
    inverse_perm = [perm.index(i) for i in range(len(perm))]
    return state.permute(inverse_perm)


def mutate_control_mask(state: Tensor, control_qubits: tuple[int, ...]) -> Tensor:
    # Create a control state mask using bitwise operations
    # Convert state to binary representation
    state_indices = torch.arange(state.numel()).view(state.shape)

    # Create a mask for control qubits being |1⟩
    control_mask = torch.zeros_like(state_indices, dtype=torch.bool)

    for qubit in control_qubits:
        # Use bitwise AND to check if the specific qubit bit is set
        control_mask |= ((state_indices >> qubit) & 1).bool()

    # Apply the mask to match exact control qubit states
    control_mask = control_mask == len(control_qubits)

    return control_mask
