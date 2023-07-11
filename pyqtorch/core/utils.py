# Copyright 2022 PyQ Development Team

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from string import ascii_letters as ABC
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

torch.set_default_dtype(torch.float64)

ABC_ARRAY: NDArray = np.array(list(ABC))


IMAT = torch.eye(2, dtype=torch.cdouble)
XMAT = torch.tensor([[0, 1], [1, 0]], dtype=torch.cdouble)
YMAT = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cdouble)
ZMAT = torch.tensor([[1, 0], [0, -1]], dtype=torch.cdouble)
SMAT = torch.tensor([[1, 0], [0, 1j]], dtype=torch.cdouble)
SDAGGERMAT = torch.tensor([[1, 0], [0, -1j]], dtype=torch.cdouble)
TMAT = torch.tensor([[1, 0], [0, torch.exp(torch.tensor(1.0j * torch.pi / 4))]])
SWAPMAT = torch.tensor(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.cdouble
)
CSWAPMAT = torch.tensor(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=torch.cdouble,
)
HMAT = 1 / torch.sqrt(torch.tensor(2)) * torch.tensor([[1, 1], [1, -1]], dtype=torch.cdouble)


OPERATIONS_DICT = {
    "I": IMAT,
    "X": XMAT,
    "Y": YMAT,
    "Z": ZMAT,
    "S": SMAT,
    "SDAGGER": SDAGGERMAT,
    "T": TMAT,
    "H": HMAT,
    "SWAP": SWAPMAT,
    "CSWAP": CSWAPMAT,
}


def _apply_gate(
    state: torch.Tensor,
    mat: torch.Tensor,
    qubits: Any,
    N_qubits: int,
) -> torch.Tensor:
    """
    Apply a gate represented by its matrix `mat` to the quantum state
    `state`

    Inputs:
    - state (torch.Tensor): input state of shape [2] * N_qubits + [batch_size]
    - mat (torch.Tensor): the matrix representing the gate
    - qubits (list, tuple, array): iterator containing the qubits
    the gate is applied to
    - N_qubits: the total number of qubits of the system

    Output:
    - state (torch.Tensor): the quantum state after application of the gate.
    Same shape as `ìnput_state`
    """
    mat = mat.reshape([2] * len(qubits) * 2)
    mat_dims = list(range(len(qubits), 2 * len(qubits)))
    state_dims = [N_qubits - i - 1 for i in list(qubits)]
    axes = (mat_dims, state_dims)

    state = torch.tensordot(mat, state, dims=axes)
    inv_perm = torch.argsort(
        torch.tensor(
            state_dims + [j for j in range(N_qubits + 1) if j not in state_dims], dtype=torch.int
        )
    )
    state = torch.permute(state, tuple(inv_perm))
    return state


def _apply_einsum_gate(
    state: torch.Tensor, mat: torch.Tensor, qubits: Any, N_qubits: int
) -> torch.Tensor:
    """
    Same as `apply_gate` but with the `torch.einsum` function

    Inputs:
    - state (torch.Tensor): input state of shape [2] * N_qubits + [batch_size]
    - mat (torch.Tensor): the matrix representing the gate
    - qubits (list, tuple, array): Sized iterator containing the qubits
    the gate is applied to
    - N_qubits: the total number of qubits of the system

    Output:
    - state (torch.Tensor): the quantum state after application of the gate.
    Same shape as `ìnput_state`
    """
    mat = mat.reshape([2] * len(qubits) * 2)
    qubits = N_qubits - 1 - np.array(qubits)

    state_indices = ABC_ARRAY[0 : N_qubits + 1]
    # Create new indices for the matrix indices
    mat_indices = ABC_ARRAY[N_qubits + 2 : N_qubits + 2 + 2 * len(qubits)]
    mat_indices[len(qubits) : 2 * len(qubits)] = state_indices[qubits]

    # Create the new state indices: same as input states but
    # modified affected qubits
    new_state_indices = state_indices.copy()
    new_state_indices[qubits] = mat_indices[0 : len(qubits)]

    # Transform the arrays into strings
    state_indices = "".join(list(state_indices))  # type: ignore
    new_state_indices = "".join(list(new_state_indices))  # type: ignore
    mat_indices = "".join(list(mat_indices))  # type: ignore

    einsum_indices = f"{mat_indices},{state_indices}->{new_state_indices}"

    state = torch.einsum(einsum_indices, mat, state)

    return state


def _apply_batch_gate(
    state: torch.Tensor, mat: torch.Tensor, qubits: Any, N_qubits: int, batch_size: int
) -> torch.Tensor:
    """
    Apply a batch execution of gates to a circuit.
    Given an tensor of states [state_0, ... state_b] and
    an tensor of gates [G_0, ... G_b] it will return the
    tensor [G_0 * state_0, ... G_b * state_b]. All gates
    are applied to the same qubit.

    Inputs:
    - state (torch.Tensor): input state of shape [2] * N_qubits + [batch_size]
    - mat (torch.Tensor): the tensor representing the gates. The last dimension
    is the batch dimension. It has to be the sam eas the last dimension of
    `state`
    - qubits (list, tuple, array): Sized iterator containing the qubits
    the gate is applied to
    - N_qubits (int): the total number of qubits of the system
    - batch_size (int): the size of the batch

    Output:
    - state (torch.Tensor): the quantum state after application of the gate.
    Same shape as `input_state`
    """
    mat = mat.reshape([2] * len(qubits) * 2 + [batch_size])
    qubits = np.array(N_qubits - 1 - np.array(qubits), dtype=np.int64)

    state_indices = ABC_ARRAY[0 : N_qubits + 1].copy()
    mat_indices = ABC_ARRAY[N_qubits + 2 : N_qubits + 2 + 2 * len(qubits) + 1].copy()
    mat_indices[len(qubits) : 2 * len(qubits)] = state_indices[qubits]
    mat_indices[-1] = state_indices[-1]

    new_state_indices = state_indices.copy()
    new_state_indices[qubits] = mat_indices[0 : len(qubits)]

    state_indices = "".join(list(state_indices))  # type: ignore
    new_state_indices = "".join(list(new_state_indices))  # type: ignore
    mat_indices = "".join(list(mat_indices))  # type: ignore

    einsum_indices = f"{mat_indices},{state_indices}->{new_state_indices}"

    state = torch.einsum(einsum_indices, mat, state)

    return state
