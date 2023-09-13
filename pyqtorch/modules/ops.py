from __future__ import annotations

import torch


def _vmap_gate(
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
