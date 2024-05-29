from __future__ import annotations

import numpy as np
import pytest
import torch

from pyqtorch import I, Sequence, X, Y, Z


@pytest.mark.parametrize("n_qubits", [3, 5, 7])
@pytest.mark.parametrize("op0", [X, Y, Z])
@pytest.mark.parametrize("op1", [X, Y, Z])
def test_block_to_tensor_support(n_qubits: int, op0: X | Y | Z, op1: X | Y | Z) -> None:
    mat0 = op0(0).tensor()
    mat1 = op1(0).tensor()
    IMAT = I(0).tensor()
    # breakpoint()

    possible_targets = list(range(n_qubits - 1))
    target = np.random.choice(possible_targets)

    qubit_support = [target, n_qubits - 1]
    np.random.shuffle(qubit_support)

    block = Sequence([op0(qubit_support[0]), op1(qubit_support[1])])

    mat_small = block.tensor(n_qubits=1)
    mat_large = block.tensor(n_qubits=n_qubits)

    if qubit_support[0] < qubit_support[1]:
        exact_small = torch.kron(mat0, mat1)
    else:
        exact_small = torch.kron(mat1, mat0)

    kron_list = [IMAT for i in range(n_qubits)]
    kron_list[qubit_support[0]] = mat0
    kron_list[qubit_support[1]] = mat1

    exact_large = kron_list[0]
    for i in range(n_qubits - 1):
        exact_large = torch.kron(exact_large, kron_list[i + 1])

    assert torch.allclose(mat_small, exact_small)
    assert torch.allclose(mat_large, exact_large)
