from __future__ import annotations

import torch
from functools import lru_cache
from typing import Any, Optional, Tuple

from pyqtorch.converters.store_ops import ops_cache, store_operation
from pyqtorch.core.utils import _apply_gate


def hamiltonian_evolution(
    H: torch.Tensor,
    state: torch.Tensor,
    t: torch.Tensor,
    qubits: Any,
    N_qubits: int,
    n_steps: int = 100,
) -> torch.Tensor:
    """A function to perform time-evolution according to the generator `H` acting on a
    `N_qubits`-sized input `state`, for a duration `t`. See also tutorials for more information
    on how to use this gate.

    Args:
        H (torch.Tensor): the dense matrix representing the Hamiltonian,
            provided as a `Tensor` object with shape
            `(N_0,N_1,...N_(N**2),batch_size)`, i.e. the matrix is reshaped into
            the list of its rows
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        t (torch.Tensor): the evolution time, real for default unitary evolution
        qubits (Any): The qubits support where the H evolution is applied
        N_qubits (int): The number of qubits
        n_steps (int, optional): The number of steps to divide the time interval
            in. Defaults to 100.

    Returns:
        torch.Tensor: replaces state with the evolved state according to the
            instructions above (save a copy of `state` if you need further
            processing on it)
    """

    if ops_cache.enabled:
        store_operation("hevo", qubits, param=t)

    # batch_size = len(t)
    # #permutation = [N_qubits - q - 1 for q in qubits]
    # permutation = [N_qubits - q - 1 for q in range(N_qubits) if q not in qubits]
    # permutation += [N_qubits - q - 1 for q in qubits]
    # permutation.append(N_qubits)
    # inverse_permutation = list(np.argsort(permutation))

    # new_dim = [2] * (N_qubits - len(qubits)) + [batch_size] + [2**len(qubits)]
    # state = state.permute(*permutation).reshape((2**len(qubits), -1))

    h = t.reshape((1, -1)) / n_steps
    for _ in range(N_qubits - 1):
        h = h.unsqueeze(0)

    h = h.expand_as(state)

    # h = h.expand(2**len(qubits), -1, 2**(N_qubits - len(qubits))).reshape((2**len(qubits), -1))
    for _ in range(n_steps):
        k1 = -1j * _apply_gate(state, H, qubits, N_qubits)
        k2 = -1j * _apply_gate(state + h / 2 * k1, H, qubits, N_qubits)
        k3 = -1j * _apply_gate(state + h / 2 * k2, H, qubits, N_qubits)
        k4 = -1j * _apply_gate(state + h * k3, H, qubits, N_qubits)
        # print(state.shape)
        # k1 = -1j * torch.matmul(H, state)
        # k2 = -1j * torch.matmul(H, state + h/2 * k1)
        # k3 = -1j * torch.matmul(H, state + h/2 * k2)
        # k4 = -1j * torch.matmul(H, state + h * k3)

        state += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return state  # .reshape([2]*N_qubits + [batch_size]).permute(*inverse_permutation)


@lru_cache(maxsize=256)
def diagonalize(H: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Diagonalizes an Hermitian Hamiltonian, returning eigenvalues and eigenvectors.
    First checks if it's already diagonal, and second checks if H is real.
    """

    def is_diag(H: torch.Tensor) -> bool:
        return len(torch.abs(torch.triu(H, diagonal=1)).to_sparse().coalesce().values()) == 0

    def is_real(H: torch.Tensor) -> bool:
        return len(torch.imag(H).to_sparse().coalesce().values()) == 0

    if is_diag(H):
        # Skips diagonalization
        eig_values = torch.diagonal(H)
        eig_vectors = None
    else:
        if is_real(H):
            eig_values, eig_vectors = torch.linalg.eigh(H.real)
            eig_values = eig_values.to(torch.cdouble)
            eig_vectors = eig_vectors.to(torch.cdouble)
        else:
            eig_values, eig_vectors = torch.linalg.eigh(H)

    return eig_values, eig_vectors


def hamiltonian_evolution_eig(
    H: torch.Tensor,
    state: torch.Tensor,
    t: torch.Tensor,
    qubits: Any,
    N_qubits: int,
) -> torch.Tensor:
    """A function to perform time-evolution according to the generator `H` acting on a
    `N_qubits`-sized input `state`, for a duration `t`. See also tutorials for more information
    on how to use this gate. Uses exact diagonalization.

    Args:
        H (torch.Tensor): the dense matrix representing the Hamiltonian,
            provided as a `Tensor` object with shape
            `(N_0,N_1,...N_(N**2),batch_size)`, i.e. the matrix is reshaped into
            the list of its rows
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        t (torch.Tensor): the evolution time, real for default unitary evolution
        qubits (Any): The qubits support where the H evolution is applied
        N_qubits (int): The number of qubits

    Returns:
        torch.Tensor: replaces state with the evolved state according to the
            instructions above (save a copy of `state` if you need further processing on it)
    """

    batch_size_s = state.size()[-1]
    batch_size_t = len(t)

    t_evo = torch.zeros(batch_size_s).to(torch.cdouble)

    if batch_size_t >= batch_size_s:
        t_evo = t[:batch_size_s]
    else:
        if batch_size_t == 1:
            t_evo[:] = t[0]
        else:
            t_evo[:batch_size_t] = t

    if ops_cache.enabled:
        store_operation("hevo", qubits, param=t)

    eig_values, eig_vectors = diagonalize(H)

    if eig_vectors is None:
        for i, t_val in enumerate(t_evo):
            # Compute e^(-i H t)
            evol_operator = torch.diag(torch.exp(-1j * eig_values * t_val))
            state[..., [i]] = _apply_gate(state[..., [i]], evol_operator, qubits, N_qubits)

    else:
        for i, t_val in enumerate(t_evo):
            # Compute e^(-i D t)
            eig_exp = torch.diag(torch.exp(-1j * eig_values * t_val))
            # e^(-i H t) = V.e^(-i D t).V^\dagger
            evol_operator = torch.matmul(
                torch.matmul(eig_vectors, eig_exp),
                torch.conj(eig_vectors.transpose(0, 1)),
            )
            state[..., [i]] = _apply_gate(state[..., [i]], evol_operator, qubits, N_qubits)

    return state
