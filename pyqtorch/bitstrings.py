from __future__ import annotations

import numpy as np
from torch import Tensor


def int_to_bitstring(k: int, n_qubits: int = 0) -> str:
    """Converts an integer to its bitstring representation."""
    return format(k, "0{}b".format(n_qubits))


def all_basis(n_qubits: int) -> list:
    """Returns a list of all basis states over n_qubits."""
    return [int_to_bitstring(k, n_qubits) for k in range(2**n_qubits)]


def _basis_order(qubit_support: tuple[int]) -> list[int]:
    """
    Returns the permutation to be applied to a set of
    basis states depending on the order of the qubit support.

    Helper function to permute the rows and columns of any operator.

    TODO: There is probably a smarter way to write this function that
    avoids explicitly writing down the basis states.
    """
    n_qubits = len(qubit_support)
    ordered_support = np.argsort(qubit_support)
    ranked_support = np.argsort(ordered_support)
    basis_array = np.array([list(map(int, tuple(b))) for b in all_basis(n_qubits)])
    basis_ordered = [basis[ranked_support] for basis in basis_array]
    return [
        sum(basis[i] * (2 ** (n_qubits - (i + 1))) for i in range(n_qubits)).item()
        for basis in basis_ordered
    ]


def permute_basis(mat: Tensor, qubit_support: tuple) -> Tensor:
    """
    Takes an operator tensor and permutes the rows and
    columns according to the order of the qubit support.
    """
    perm = _basis_order(qubit_support)
    return mat[:, perm, :][perm, :, :]
