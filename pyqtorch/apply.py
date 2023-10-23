from __future__ import annotations

from string import ascii_letters as ABC

import numpy as np
import torch
from numpy.typing import NDArray

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
    inverse_permutation = tuple(
        torch.argsort(
            torch.tensor(
                qubits + [j for j in range(n_qubits + 1) if j not in qubits], dtype=torch.int
            )
        )
    )
    return torch.permute(state, inverse_permutation)


def _apply_batch_gate(
    state: torch.Tensor,
    operator: torch.Tensor,
    qubits: list[int],
    n_qubits: int,
    batch_size: int = None,
) -> torch.Tensor:
    if batch_size is None:
        batch_size = state.size(-1)
    n_support = len(qubits)
    operator = operator.view([2] * n_support * 2 + [batch_size])
    state_indices = ABC_ARRAY[0 : n_qubits + 1].copy()
    operator_indices = ABC_ARRAY[n_qubits + 2 : n_qubits + 2 + 2 * n_support + 1].copy()
    operator_indices[n_support : 2 * n_support] = state_indices[qubits]
    operator_indices[-1] = state_indices[-1]
    new_state_indices = state_indices.copy()
    new_state_indices[qubits] = operator_indices[0:n_support]
    operator_indices, state_indices, new_state_indices = list(
        map(lambda expr: "".join(list(expr)), [operator_indices, state_indices, new_state_indices])
    )
    return torch.einsum(f"{operator_indices},{state_indices}->{new_state_indices}", operator, state)


def _vmap_apply_gate(
    state: torch.Tensor,
    operator: torch.Tensor,
    qubits: list[int] | tuple[int],
    n_qubits: int,
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
        in_dims=(len(state.size()) - 1, len(operator.size()) - 1),
        out_dims=len(state.size()) - 1,
    )(state, operator)


DEFAULT_APPLY_FN = _apply_batch_gate
