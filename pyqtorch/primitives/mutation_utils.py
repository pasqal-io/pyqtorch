from __future__ import annotations

from torch import Tensor


def mutate_separate_target(
    state: Tensor, target_qubit: tuple[int, ...]
) -> tuple[list[int], Tensor]:
    """Create a tensor separating the target components
    for a single-qubit gate for mutating an input state-vector.

    Args:
        state (Tensor): Input state.
        target_qubit (tuple[int, ...]): Target indices.

    Returns:
        tuple[int list[int], Tensor]: The permutation indices with an intermediate state
            with separated target qubit.
    """
    n_qubits = len(state.shape) - 1
    perm = list(range(n_qubits + 1))
    perm = list(target_qubit) + list(filter(lambda x: x not in target_qubit, perm))

    # Transpose the state
    state = state.permute(perm)

    # Reshape to separate the target qubit
    state = state.reshape(2 * len(target_qubit), -1)
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


def mutate_control_slice(n_qubits: int, control_qubits: tuple[int, ...]) -> tuple:
    """Create a slice for mutable controlled operations.

    Args:
        n_qubits (int): Number of qubits
        control_qubits (tuple[int, ...]): Control indices.

    Returns:
        tuple: slices to select controlled states.
    """
    selector: list = [slice(None)] * n_qubits
    for qubit in control_qubits:
        selector[qubit] = 1
    return tuple(selector)
