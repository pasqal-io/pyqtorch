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

from pyqtorch.operator import Operator
from pyqtorch.utils import ApplyFn, reverse_permutation

ABC_ARRAY: NDArray = np.array(list(ABC))


def _apply_gate(
    state: torch.Tensor,
    operator: torch.Tensor,
    qubits: list[int],
    n_qubits: int,
) -> torch.Tensor:
    operator = operator.view([2] * len(qubits) * 2)
    operator_indices = list(range(len(qubits), 2 * len(qubits)))
    dims = (operator_indices, qubits)
    state = torch.tensordot(operator, state, dims=dims)
    return reverse_permutation(state, qubits, n_qubits)


def _apply_batch_gate(
    state: torch.Tensor,
    operator: torch.Tensor,
    qubits: list[int],
    n_qubits: int,
    batch_size: int,
) -> torch.Tensor:
    operator = operator.view([2] * len(qubits) * 2 + [batch_size])
    state_indices = ABC_ARRAY[0 : n_qubits + 1].copy()
    operator_indices = ABC_ARRAY[n_qubits + 2 : n_qubits + 2 + 2 * len(qubits) + 1].copy()
    operator_indices[len(qubits) : 2 * len(qubits)] = state_indices[qubits]
    operator_indices[-1] = state_indices[-1]
    new_state_indices = state_indices.copy()
    new_state_indices[qubits] = operator_indices[0 : len(qubits)]
    state_indices = "".join(list(state_indices))
    new_state_indices = "".join(list(new_state_indices))
    operator_indices = "".join(list(operator_indices))
    einsum_indices = f"{operator_indices},{state_indices}->{new_state_indices}"
    return torch.einsum(einsum_indices, operator, state)


def _vmap_apply_gate(
    state: torch.Tensor,
    mat: torch.Tensor,
    qubits: list[int] | tuple[int],
    n_qubits: int,
    batch_size: int,
) -> torch.Tensor:
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
    gates: Tuple[Operator] | list[Operator],
    n_qubits: int,
) -> torch.Tensor:
    qubits_list: list[Tuple] = [g.qubits for g in gates]
    mats = [g.unitary(thetas) for g in gates]

    return reduce(
        lambda state, inputs: _apply_gate(state, *inputs, n_qubits=n_qubits),  # type: ignore
        zip(mats, qubits_list),
        state,
    )


DEFAULT_APPLY_FN = _apply_batch_gate
