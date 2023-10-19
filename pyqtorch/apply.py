# Copyright 2023 pyqtorch Development Team

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

from functools import reduce
from string import ascii_letters as ABC
from typing import Tuple

import numpy as np
import torch
from numpy.typing import NDArray

from pyqtorch.modules.abstract import AbstractOperator

ABC_ARRAY: NDArray = np.array(list(ABC))


def _apply_gate(
    state: torch.Tensor,
    mat: torch.Tensor,
    qubits: list[int] | tuple[int],
    n_qubits: int,
) -> torch.Tensor:
    """
    Apply the matrix 'mat' corresponding to a gate to `state`.

    Arguments:
        state (torch.Tensor): input state of shape [2] * N_qubits + [batch_size]
        mat (torch.Tensor): the matrix representing the gate
        qubits (list, tuple, array): iterator containing the qubits
        the gate is applied to
        n_qubits (int): the total number of qubits of the system

    Returns:
     state (torch.Tensor): the quantum state after application of the gate.
            Same shape as `ìnput_state`
    """

    mat = mat.view([2] * len(qubits) * 2)
    mat_dims = list(range(len(qubits), 2 * len(qubits)))
    qubits = list(qubits)
    axes = (mat_dims, qubits)
    state = torch.tensordot(mat, state, dims=axes)
    inv_perm = torch.argsort(
        torch.tensor(qubits + [j for j in range(n_qubits + 1) if j not in qubits], dtype=torch.int)
    )
    state = torch.permute(state, tuple(inv_perm))
    return state


def _apply_einsum_gate(
    state: torch.Tensor, mat: torch.Tensor, qubits: list[int] | tuple[int], N_qubits: int
) -> torch.Tensor:
    """
    Same as `apply_gate` but with the `torch.einsum` function

    Arguments:
        state (torch.Tensor): input state of shape [2] * N_qubits + [batch_size]
        mat (torch.Tensor): the matrix representing the gate
        qubits (list, tuple, array): Sized iterator containing the qubits
                                     the gate is applied to
        N_qubits (int): the total number of qubits of the system

    Returns:
        state (torch.Tensor): The state after application of the gate.
                              Same shape as `ìnput_state`
    """
    mat = mat.reshape([2] * len(qubits) * 2)
    state_indices = ABC_ARRAY[0 : N_qubits + 1]
    qubits = list(qubits)
    # Create new indices for the matrix indices
    mat_indices = ABC_ARRAY[N_qubits + 2 : N_qubits + 2 + 2 * len(qubits)]
    mat_indices[len(qubits) : 2 * len(qubits)] = state_indices[qubits]

    # Create the new state indices: same as input states but
    # modified affected qubits
    new_state_indices = state_indices.copy()
    new_state_indices[qubits] = mat_indices[0 : len(qubits)]

    # Transform the arrays into strings
    state_indices = "".join(list(state_indices))
    new_state_indices = "".join(list(new_state_indices))
    mat_indices = "".join(list(mat_indices))

    einsum_indices = f"{mat_indices},{state_indices}->{new_state_indices}"

    state = torch.einsum(einsum_indices, mat, state)

    return state


def _apply_batch_gate(
    state: torch.Tensor,
    mat: torch.Tensor,
    qubits: list[int] | tuple[int],
    n_qubits: int,
    batch_size: int,
) -> torch.Tensor:
    """
    Apply a batch execution of gates to a circuit.
    Given an tensor of states [state_0, ... state_b] and
    an tensor of gates [G_0, ... G_b] it will return the
    tensor [G_0 * state_0, ... G_b * state_b]. All gates
    are applied to the same qubit.

    Arguments:
        state (torch.Tensor): input state of shape [2] * N_qubits + [batch_size]
        mat (torch.Tensor): the tensor representing the gates.
            The last dimension is the batch dimension.
            It has to be the sam eas the last dimension of `state`
        qubits (list, tuple, array): Sized iterator containing the qubits
                                        the gate is applied to.
        n_qubits (int): the total number of qubits of the system
        batch_size (int): the size of the batch

    Returns:
        torch.Tensor: The state after application of the gate.
                                Same shape as `input_state`.
    Examples:
    ```python exec="on" source="above" result="json"
    import torch
    import pyqtorch.modules as pyq

    state = pyq.zero_state(n_qubits=2)
    print(state)  #tensor([[[1.+0.j],[0.+0.j]],[[0.+0.j],[0.+0.j]]], dtype=torch.complex128)
    ```
    """
    mat = mat.view([2] * len(qubits) * 2 + [batch_size])

    state_indices = ABC_ARRAY[0 : n_qubits + 1].copy()
    qubits = list(qubits)
    mat_indices = ABC_ARRAY[n_qubits + 2 : n_qubits + 2 + 2 * len(qubits) + 1].copy()
    mat_indices[len(qubits) : 2 * len(qubits)] = state_indices[qubits]
    mat_indices[-1] = state_indices[-1]

    new_state_indices = state_indices.copy()
    new_state_indices[qubits] = mat_indices[0 : len(qubits)]

    state_indices = "".join(list(state_indices))
    new_state_indices = "".join(list(new_state_indices))
    mat_indices = "".join(list(mat_indices))

    einsum_indices = f"{mat_indices},{state_indices}->{new_state_indices}"

    state = torch.einsum(einsum_indices, mat, state)

    return state


def _vmap_operator(
    state: torch.Tensor,
    mat: torch.Tensor,
    qubits: list[int] | tuple[int],
    n_qubits: int,
    batch_size: int,
) -> torch.Tensor:
    """
    Vmap a batched gate over a batched state and
    apply the matrix 'mat' to `state`.

    Arguments:
        state (torch.Tensor): input state of shape [2] * N_qubits + [batch_size]
        mat (torch.Tensor): the matrix representing the gate
        qubits (list, tuple, array): iterator containing the qubits
        the gate is applied to
        n_qubits (int): the total number of qubits of the system
        batch

    Returns:
     state (torch.Tensor): the quantum state after application of the gate.
            Same shape as `ìnput_state`
    """

    def _apply(
        state: torch.Tensor,
        mat: torch.Tensor,
        qubits: list[int] | tuple[int],
        n_qubits: int,
    ) -> torch.Tensor:
        mat = mat.view([2] * len(qubits) * 2)
        mat_dims = list(range(len(qubits), 2 * len(qubits)))
        state_dims = list(qubits)
        axes = (mat_dims, state_dims)
        state = torch.tensordot(mat, state, dims=axes)
        inv_perm = torch.argsort(
            torch.tensor(
                state_dims + [j for j in range(n_qubits) if j not in state_dims], dtype=torch.int
            )
        )
        state = torch.permute(state, tuple(inv_perm))
        return state

    return torch.vmap(
        lambda s, m: _apply(state=s, mat=m, qubits=qubits, n_qubits=n_qubits),
        in_dims=(len(state.size()) - 1, len(mat.size()) - 1),
        out_dims=len(state.size()) - 1,
    )(state, mat)


def _apply_parallel(
    state: torch.Tensor,
    thetas: torch.Tensor,
    gates: Tuple[AbstractOperator] | list[AbstractOperator],
    n_qubits: int,
) -> torch.Tensor:
    qubits_list: list[Tuple] = [g.qubits for g in gates]
    mats = [g.unitary(thetas) for g in gates]

    return reduce(
        lambda state, inputs: _apply_gate(state, *inputs, n_qubits=n_qubits),  # type: ignore
        zip(mats, qubits_list),
        state,
    )


def overlap(state: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    n_qubits = len(state.size()) - 1
    batch_size = state.size()[-1]
    state = state.reshape((2**n_qubits, batch_size))
    other = other.reshape((2**n_qubits, batch_size))
    res = []
    for i in range(batch_size):
        ovrlp = torch.real(torch.sum(torch.conj(state[:, i]) * other[:, i]))
        res.append(ovrlp)
    return torch.stack(res)
