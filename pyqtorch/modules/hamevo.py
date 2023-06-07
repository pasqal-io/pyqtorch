from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional, Tuple

import torch
from torch.nn import Module

from pyqtorch.core.utils import _apply_batch_gate
from pyqtorch.modules.utils import is_diag, is_real

BATCH_DIM = 2


class HamEvo(torch.nn.Module):
    def __init__(
        self, H: torch.Tensor, t: torch.Tensor, qubits: Any, n_qubits: int, n_steps: int = 100
    ):
        """
                Represents Hamiltonian evolution over a specified time period.

                Args:
                    H (torch.Tensor): The Hamiltonian operator.
                    t (torch.Tensor): The time period for evolution.
                    qubits (Any): The target qubits for the evolution.
                    n_qubits (int): The total number of qubits in the system.
                    n_steps (int, optional): The number of steps for numerical integration. Defaults to 100.
                """
        super().__init__()
        self.H: torch.Tensor
        self.t: torch.Tensor
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.n_steps = n_steps
        self.register_buffer("H", H)
        self.register_buffer("t", t)

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """
               Apply Hamiltonian evolution to the input state.

               Args:
                   state (torch.Tensor): The input state.

               Returns:
                   torch.Tensor: The evolved state.

               Examples:
                   ```python exec="on" source="above" result="json"
                   H = torch.tensor([[1, 0], [0, -1]])
                   t = torch.tensor([0.5])
                   qubits = ...
                   n_qubits = 2
                   n_steps = 100
                   ham_evo = HamEvo(H, t, qubits, n_qubits, n_steps)
                   state = ...
                   evolved_state = ham_evo.apply(state)
                   '''
               """
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
        """
                Apply Hamiltonian evolution to the input state.

                Args:
                    state (torch.Tensor): The input state.

                Returns:
                    torch.Tensor: The evolved state.

                Examples:
                    ```python exec="on" source="above" result="json"
                    H = torch.tensor([[1, 0], [0, -1]])
                    t = torch.tensor([0.5])
                    qubits = ...
                    n_qubits = 2
                    n_steps = 100
                    ham_evo = HamEvo(H, t, qubits, n_qubits, n_steps)
                    state = ...
                    evolved_state = ham_evo(state)
                    '''
                """
        return self.apply(state)


@lru_cache(maxsize=256)
def diagonalize(H: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
        Diagonalizes the input Hermitian matrix.

        Args:
            H (torch.Tensor): The Hermitian matrix to be diagonalized.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing the eigenvalues and
                eigenvectors of the matrix. If the matrix is already diagonal, eigenvectors
                will be None.

        Examples:
            ```python exec="on" source="above" result="json"
            H = torch.tensor([[1, 0], [0, 2]])
            eig_values, eig_vectors = diagonalize(H)
            '''
        """

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
            eig_values, eig_vectors = diagonalize(self.H[..., i])

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


class HamEvoExp(HamEvo):
    def __init__(
        self, H: torch.Tensor, t: torch.Tensor, qubits: Any, n_qubits: int, n_steps: int = 100
    ):
        """
                Represents a Hamiltonian evolution with diagonalization.

                Args:
                    H (torch.Tensor): The Hamiltonian matrix.
                    t (torch.Tensor): The time values for evolution.
                    qubits (Any): The target qubits for the evolution.
                    n_qubits (int): The total number of qubits in the circuit.
                    n_steps (int, optional): The number of steps for the evolution. Defaults to 100.
                """
        super().__init__(H, t, qubits, n_qubits, n_steps)
        if len(self.H.size()) < 3:
            self.H = self.H.unsqueeze(2)
        batch_size_h = self.H.size()[BATCH_DIM]

        # Check if all hamiltonians in the batch are diagonal
        diag_check = torch.tensor([is_diag(self.H[..., i]) for i in range(batch_size_h)])
        self.batch_is_diag = bool(torch.prod(diag_check))

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """
                Apply the diagonalized Hamiltonian evolution to the input state.

                Args:
                    state (torch.Tensor): The input state.

                Returns:
                    torch.Tensor: The evolved state.

                Raises:
                    AssertionError: If the number of time steps is incompatible with the Hamiltonian.

                Examples:
                    ```python exec="on" source="above" result="json"
                    H = torch.tensor([[1, 0], [0, 2]])
                    t = torch.tensor([0.1, 0.2, 0.3])
                    qubits = [0]
                    n_qubits = 1
                    ham_evo = HamEvoEig(H, t, qubits, n_qubits)
                    state = torch.tensor([1, 0], dtype=torch.cdouble)
                    evolved_state = ham_evo.apply(state)
                    '''
                """
        batch_size_t = len(self.t)
        batch_size_h = self.H.size()[BATCH_DIM]
        t_evo = torch.zeros(batch_size_h).to(torch.cdouble)

        if batch_size_t >= batch_size_h:
            t_evo = self.t[:batch_size_h]
        else:
            if batch_size_t == 1:
                t_evo[:] = self.t[0]
            else:
                t_evo[:batch_size_t] = self.t

        if self.batch_is_diag:
            # Skips the matrix exponential for diagonal hamiltonians
            H_diagonals = torch.diagonal(self.H)
            evol_exp_arg = H_diagonals * (-1j * t_evo).view((-1, 1))
            evol_operator_T = torch.diag_embed(torch.exp(evol_exp_arg))
            evol_operator = torch.transpose(evol_operator_T, 0, -1)
        else:
            H_T = torch.transpose(self.H, 0, -1)
            evol_exp_arg = H_T * (-1j * t_evo).view((-1, 1, 1))
            evol_operator_T = torch.linalg.matrix_exp(evol_exp_arg)
            evol_operator = torch.transpose(evol_operator_T, 0, -1)

        return _apply_batch_gate(state, evol_operator, self.qubits, self.n_qubits, batch_size_h)


class HamiltonianEvolution(Module):
    def __init__(self, qubits: Any, n_qubits: int, n_steps: int = 100):
        """
                Represents a Hamiltonian evolution.

                Args:
                    qubits (Any): The target qubits for the evolution.
                    n_qubits (int): The total number of qubits in the circuit.
                    n_steps (int, optional): The number of steps for the evolution. Defaults to 100.
                """
        super().__init__()
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.n_steps = n_steps

    def forward(self, H: torch.Tensor, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
                Apply the Hamiltonian evolution to the input state.

                Args:
                    H (torch.Tensor): The Hamiltonian matrix.
                    t (torch.Tensor): The time values for evolution.
                    state (torch.Tensor): The input state.

                Returns:
                    torch.Tensor: The evolved state.

                Examples:
                    ```python exec="on" source="above" result="json"
                    H = torch.tensor([[1, 0], [0, 2]])
                    t = torch.tensor([0.1, 0.2, 0.3])
                    qubits = [0]
                    n_qubits = 1
                    ham_evo = HamiltonianEvolution(qubits, n_qubits)
                    state = torch.tensor([1, 0], dtype=torch.cdouble)
                    evolved_state = ham_evo(H, t, state)
                    '''
                """
        return self.apply(H, t, state)

    def apply(self, H: torch.Tensor, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
            Apply Hamiltonian evolution on the input state.

            Args:
                H (torch.Tensor): The Hamiltonian matrix.
                t (torch.Tensor): The time values for evolution.
                state (torch.Tensor): The input state.

            Returns:
                torch.Tensor: The evolved state.

            Raises:
                AssertionError: If the number of time steps is incompatible with the Hamiltonian.

            Examples:
                 ```python exec="on" source="above" result="json"
                H = torch.tensor([[1, 0], [0, 2]])
                t = torch.tensor([0.1, 0.2, 0.3])
                qubits = [0]
                n_qubits = 1
                ham_evo = HamiltonianEvolution(qubits, n_qubits)
                state = torch.tensor([1, 0], dtype=torch.cdouble)
                evolved_state = ham_evo.apply(H, t, state)
                '''
            """
        if len(H.size()) < 3:
            H = H.unsqueeze(2)
        batch_size_h = H.size()[BATCH_DIM]

        # Check if all hamiltonians in the batch are diagonal
        diag_check = torch.tensor([is_diag(H[..., i]) for i in range(batch_size_h)])
        batch_is_diag = bool(torch.prod(diag_check))

        batch_size_t = len(t)
        t_evo = torch.zeros(batch_size_h).to(torch.cdouble)

        if batch_size_t >= batch_size_h:
            t_evo = t[:batch_size_h]
        else:
            if batch_size_t == 1:
                t_evo[:] = t[0]
            else:
                t_evo[:batch_size_t] = t

        if batch_is_diag:
            # Skips the matrix exponential for diagonal hamiltonians
            H_diagonals = torch.diagonal(H)
            evol_exp_arg = H_diagonals * (-1j * t_evo).view((-1, 1))
            evol_operator_T = torch.diag_embed(torch.exp(evol_exp_arg))
            evol_operator = torch.transpose(evol_operator_T, 0, -1)
        else:
            H_T = torch.transpose(H, 0, -1)
            evol_exp_arg = H_T * (-1j * t_evo).view((-1, 1, 1))
            evol_operator_T = torch.linalg.matrix_exp(evol_exp_arg)
            evol_operator = torch.transpose(evol_operator_T, 0, -1)

        return _apply_batch_gate(state, evol_operator, self.qubits, self.n_qubits, batch_size_h)
