from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional, Tuple

import torch

from pyqtorch.core.utils import _apply_batch_gate

BATCH_DIM = 2


class HamEvo(torch.nn.Module):
    def __init__(
        self, H: torch.Tensor, t: torch.Tensor, qubits: Any, n_qubits: int, n_steps: int = 100
    ):
        super().__init__()
        self.H: torch.Tensor
        self.t: torch.Tensor
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.n_steps = n_steps
        self.register_buffer("H", H)
        self.register_buffer("t", t)

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.size(-1)
        h = self.t.reshape((1, -1)) / self.n_steps
        for _ in range(self.n_qubits - 1):
            h = h.unsqueeze(0)

        h = h.expand_as(state)
        _state = state.clone()
        for _ in range(self.n_steps):
            k1 = -1j * _apply_batch_gate(_state, self.H, self.qubits, self.n_qubits, batch_size)
            k2 = -1j * _apply_batch_gate(
                _state + h / 2 * k1, self.H, self.qubits, self.n_qubits, batch_size
            )
            k3 = -1j * _apply_batch_gate(
                _state + h / 2 * k2, self.H, self.qubits, self.n_qubits, batch_size
            )
            k4 = -1j * _apply_batch_gate(
                _state + h * k3, self.H, self.qubits, self.n_qubits, batch_size
            )
            _state += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return _state

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.apply(state)


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


class HamEvoEig(HamEvo):
    def __init__(
        self, H: torch.Tensor, t: torch.Tensor, qubits: Any, n_qubits: int, n_steps: int = 100
    ):
        super().__init__(H, t, qubits, n_qubits, n_steps)
        if len(self.H.size()) < 3:
            self.H = self.H.unsqueeze(2)
        batch_size_h = self.H.size()[BATCH_DIM]
        self.l_vec = []
        self.l_val = []
        for i in range(batch_size_h):
            eig_values, eig_vectors = diagonalize(self.H[..., [i]])
            self.l_vec.append(eig_vectors)
            self.l_val.append(eig_values)

    def apply(self, state: torch.Tensor) -> torch.Tensor:

        batch_size_t = len(self.t)
        batch_size_h = self.H.size()[BATCH_DIM]
        t_evo = torch.zeros(batch_size_h).to(torch.cdouble)
        evol_operator = torch.zeros(self.H.size()).to(torch.cdouble)

        if batch_size_t >= batch_size_h:
            t_evo = self.t[:batch_size_h]
        else:
            if batch_size_t == 1:
                t_evo[:] = self.t[0]
            else:
                t_evo[:batch_size_t] = self.t

        for i in range(batch_size_h):
            eig_values, eig_vectors = self.l_val[i], self.l_vec[i]
            eig_values = eig_values.flatten(0)

            if eig_vectors is None:
                # Compute e^(-i H t)
                evol_operator[..., i] = torch.diag(torch.exp(-1j * eig_values * t_evo[i]))

            else:
                # Compute e^(-i D t)
                eig_exp = torch.diag(torch.exp(-1j * eig_values * t_evo[i]))
                # e^(-i H t) = V.e^(-i D t).V^\dagger
                evol_operator[..., i] = torch.matmul(
                    torch.matmul(eig_vectors, eig_exp),
                    torch.conj(eig_vectors.transpose(0, 1)),
                )

        return _apply_batch_gate(state, evol_operator, self.qubits, self.n_qubits, batch_size_h)
