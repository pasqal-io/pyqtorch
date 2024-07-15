from __future__ import annotations

import numpy as np
import torch


def _basis_order(qs: tuple[int]) -> list[int]:
    """
    Returns the permutation to be applied to a set of
    basis states depending on the order of the qubit support.

    Helper function to permute the rows and columns of any operator.
    """
    n_qubits = len(qs)
    qs_ordered = np.argsort(qs)
    basis_strings = [format(k, "0{}b".format(n_qubits)) for k in range(2**n_qubits)]
    basis_list = np.array([list(map(int, tuple(basis))) for basis in basis_strings])
    basis_ordered = [basis[qs_ordered] for basis in basis_list]
    return [
        sum(basis[i] * (2 ** (n_qubits - (i + 1))) for i in range(n_qubits)).item()
        for basis in basis_ordered
    ]


def permute_basis(mat: torch.Tensor, qs: tuple) -> torch.Tensor:
    """
    Takes an operator tensor and permutes the rows and
    columns according to the order of the qubit support.
    """
    perm = _basis_order(qs)
    return mat[:, perm, :][perm, :, :]
