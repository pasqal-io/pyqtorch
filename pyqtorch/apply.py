from __future__ import annotations

from string import ascii_letters as ABC
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from pyqtorch.utils import ApplyFn, Operator, State

ABC_ARRAY: NDArray = np.array(list(ABC))


def _apply_tensordot(
    state: State,
    operator: Operator,
    qubits: list[int],
    n_qubits: int,
) -> State:
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


def _apply_einsum(
    state: State,
    operator: Operator,
    qubits: list[int],
    n_qubits: int,
    batch_size: int = None,
) -> State:
    if batch_size is None:
        batch_size = state.size(-1)
    n_support = len(qubits)
    operator = operator.view([2] * n_support * 2 + [operator.size(-1)])
    in_state_dims = ABC_ARRAY[0 : n_qubits + 1].copy()
    op_dims = ABC_ARRAY[n_qubits + 2 : n_qubits + 2 + 2 * n_support + 1].copy()
    op_dims[n_support : 2 * n_support] = in_state_dims[qubits]
    op_dims[-1] = in_state_dims[-1]
    out_state_dims = in_state_dims.copy()
    out_state_dims[qubits] = op_dims[0:n_support]
    op_dims, in_state_dims, out_state_dims = list(
        map(lambda e: "".join(list(e)), [op_dims, in_state_dims, out_state_dims])
    )
    return torch.einsum(f"{op_dims},{in_state_dims}->{out_state_dims}", operator, state)


def _apply_vmap(
    state: State,
    operator: Operator,
    qubits: list[int] | tuple[int],
    n_qubits: int,
) -> State:
    def _apply(
        state: State,
        operator: Operator,
        qubits: list[int] | tuple[int],
        n_qubits: int,
    ) -> State:
        operator = operator.view([2] * len(qubits) * 2)
        mat_dims = list(range(len(qubits), 2 * len(qubits)))
        state_dims = list(qubits)
        axes = (mat_dims, state_dims)
        state = torch.tensordot(operator, state, dims=axes)
        inv_perm = torch.argsort(
            torch.tensor(
                state_dims + [j for j in range(n_qubits) if j not in state_dims], dtype=torch.int
            )
        )
        state = torch.permute(state, tuple(inv_perm))
        return state

    return torch.vmap(
        lambda s, m: _apply(state=s, operator=m, qubits=qubits, n_qubits=n_qubits),
        in_dims=(len(state.size()) - 1, len(operator.size()) - 1),
        out_dims=len(state.size()) - 1,
    )(state, operator)


DEFAULT_APPLY_FN = _apply_einsum

APPLY_FN_DICT = {ApplyFn.VMAP: _apply_vmap, ApplyFn.EINSUM: _apply_einsum}


def apply(
    state: State, operator: Operator, qubit_support: list[int], apply_fn: ApplyFn = ApplyFn.EINSUM
) -> State:
    fn: Any = APPLY_FN_DICT[apply_fn]
    return fn(state, operator, qubit_support, len(state.size()) - 1)
