from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Any, Optional, Tuple, Union

import torch
from torch.nn import Module

from pyqtorch.core.utils import _apply_batch_gate
from pyqtorch.modules.utils import is_diag, is_real

BATCH_DIM = 2


class HamEvo(torch.nn.Module):
    """
    Base class for Hamiltonian evolution classes, performing the evolution using RK4 method.

    Args:
        H (tensor): Hamiltonian tensor.
        t (tensor): Time tensor.
        qubits (Any): Qubits for operation.
        n_qubits (int): Number of qubits.
        n_steps (int): Number of steps to be performed in RK4-based evolution. Defaults to 100.


    """

    def __init__(
        self, H: torch.Tensor, t: torch.Tensor, qubits: Any, n_qubits: int, n_steps: int = 100
    ):
        super().__init__()
        self.H: torch.Tensor
        self.t: torch.Tensor
        self.qubits = [n_qubits - i - 1 for i in qubits]
        self.n_qubits = n_qubits
        self.n_steps = n_steps
        if H.ndim == 2:
            H = H.unsqueeze(2)
        if H.size(-1) == t.size(0) or t.size(0) == 1:
            self.register_buffer("H", H)
            self.register_buffer("t", t)
        elif H.size(-1) == 1:
            (x, y, _) = H.size()
            self.register_buffer("H", H.expand(x, y, t.size(0)))
            self.register_buffer("t", t)
        else:
            msg = "H and t batchsizes either have to match or (one of them has to) be equal to one."
            raise ValueError(msg)

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """
        Applies the Hamiltonian evolution operation on the given state using RK4 method.

        Args:
            state (tensor): Input quantum state.

        Returns:
            tensor: Output state after Hamiltonian evolution.
        """

        batch_size = max(state.size(-1), self.H.size(-1))
        if state.size(-1) == 1:
            state = state.repeat(*[1 for _ in range(len(state.size()) - 1)], batch_size)

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
        Forward pass of the module, applies the Hamiltonian evolution operation.

        Args:
            state (tensor): Input quantum state.

        Returns:
            tensor: Output state after Hamiltonian evolution.
        """

        return self.apply(state)


@lru_cache(maxsize=256)
def diagonalize(H: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Diagonalizes an Hermitian Hamiltonian, returning eigenvalues and eigenvectors.
    First checks if it's already diagonal, and second checks if H is real.
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
    """
    Class for Hamiltonian evolution operation using Eigenvalue Decomposition method.

    Args:
        H (tensor): Hamiltonian tensor
        t (tensor): Time tensor
        qubits (Any): Qubits for operation
        n_qubits (int): Number of qubits
        n_steps (int): Number of steps to be performed, defaults to 100
    """

    def __init__(
        self, H: torch.Tensor, t: torch.Tensor, qubits: Any, n_qubits: int, n_steps: int = 100
    ):
        super().__init__(H, t, qubits, n_qubits, n_steps)
        if len(self.H.size()) < 3:
            self.H = self.H.unsqueeze(2)
        batch_size_h = self.H.size()[BATCH_DIM]
        batch_size_t = self.t.size(0)

        self._eigs = []
        if batch_size_h == batch_size_t or batch_size_t == 1:
            for i in range(batch_size_h):
                eig_values, eig_vectors = diagonalize(self.H[..., i])
                self._eigs.append((eig_values, eig_vectors))
        elif batch_size_h == 1:
            eig_values, eig_vectors = diagonalize(self.H[..., 0])
            for i in range(batch_size_t):
                self._eigs.append((eig_values, eig_vectors))
        else:
            msg = "H and t batchsizes either have to match or (one of them has to) be equal to one."
            raise ValueError(msg)
        self.batch_size = max(batch_size_h, batch_size_t)

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """
        Applies the Hamiltonian evolution operation on the given state
        using Eigenvalue Decomposition method.

        Args:
            state (tensor): Input quantum state.

        Returns:
            tensor: Output state after Hamiltonian evolution.
        """

        (x, y, _) = self.H.size()
        evol_operator = torch.zeros(x, y, self.batch_size).to(torch.cdouble)
        t_evo = self.t.repeat(self.batch_size) if self.t.size(0) == 1 else self.t

        for i, (eig_values, eig_vectors) in enumerate(self._eigs):
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

        return _apply_batch_gate(state, evol_operator, self.qubits, self.n_qubits, self.batch_size)


class HamEvoExp(HamEvo):
    """
    Class for Hamiltonian evolution operation, using matrix exponential method.

    Args:
        H (tensor): Hamiltonian tensor
        t (tensor): Time tensor
        qubits (Any): Qubits for operation
        n_qubits (int): Number of qubits
        n_steps (int): Number of steps to be performed, defaults to 100.
    """

    def __init__(
        self, H: torch.Tensor, t: torch.Tensor, qubits: Any, n_qubits: int, n_steps: int = 100
    ):
        super().__init__(H, t, qubits, n_qubits, n_steps)
        if len(self.H.size()) < 3:
            self.H = self.H.unsqueeze(2)
        batch_size_h = self.H.size()[BATCH_DIM]

        # Check if all hamiltonians in the batch are diagonal
        diag_check = torch.tensor([is_diag(self.H[..., i]) for i in range(batch_size_h)])
        self.batch_is_diag = bool(torch.prod(diag_check))

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """
        Applies the Hamiltonian evolution operation
        on the given state using matrix exponential method.

        Args:
            state (tensor): Input quantum state.

        Returns:
            tensor: Output state after Hamiltonian evolution.
        """

        batch_size_t = len(self.t)
        batch_size_h = self.H.size()[BATCH_DIM]
        t_evo = self.t

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

        batch_size = max(batch_size_h, batch_size_t)
        return _apply_batch_gate(state, evol_operator, self.qubits, self.n_qubits, batch_size)


class HamEvoType(Enum):
    """
    An Enumeration to represent types of Hamiltonian Evolution

    RK4: Hamiltonian evolution performed using the 4th order Runge-Kutta method.
    EIG: Hamiltonian evolution performed using Eigenvalue Decomposition.
    EXP: Hamiltonian evolution performed using the Exponential of the Hamiltonian.
    """

    RK4 = HamEvo
    EIG = HamEvoEig
    EXP = HamEvoExp


class HamiltonianEvolution(Module):
    """
    A module to encapsulate Hamiltonian Evolution operations.

    Performs Hamiltonian Evolution using different strategies
    such as, RK4, Eigenvalue Decomposition, and Exponential,
    Based on the 'hamevo_type' parameter.

    Attributes:
        qubits: A list of qubits to be used in the operation
        n_qubits (int): Total number of qubits
        n_steps  (int): The number of steps to be performed. Defaults to 100
        hamevo_type (Enum or str): The type of Hamiltonian evolution to be performed.
                     Must be a member of the `HamEvoType` enum or equivalent string.
                     Defaults to HamEvoExp.

    Examples:
    (1)
    # Instantiate HamiltonianEvolution with RK4 string input
        >>> hamiltonian_evolution = HamiltonianEvolution(qubits, n_qubits, 100, "RK4")
    # Use the HamiltonianEvolution instance to evolve the state
        >>> output_state = hamiltonian_evolution(H, t, state)

    (2)
    # Instantiate HamiltonianEvolution with HamEvoType. input
        >>>H_evol = HamiltonianEvolution(qubits, n_qubits, 100, HamEvoType.Eig)
    # Use the HamiltonianEvolution instance to evolve the state
        >>>output = H_evol(H, t, state) # Use the HamiltonianEvolution instance to evolve the state

    """

    def __init__(
        self,
        qubits: Any,
        n_qubits: int,
        n_steps: int = 100,
        hamevo_type: Union[HamEvoType, str] = HamEvoType.EXP,
    ):
        super().__init__()
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.n_steps = n_steps

        # Handles the case where the Hamiltonian Evolution type is provided as a string
        if isinstance(hamevo_type, str):
            try:
                hamevo_type = HamEvoType[hamevo_type.upper()]
            except KeyError:
                allowed = [e.name for e in HamEvoType]
                raise ValueError(
                    f"Invalid Hamiltonian Evolution type: {hamevo_type}. Expected from: {allowed}"
                )

        self.hamevo_type = hamevo_type

    def get_hamevo_instance(self, H: torch.Tensor, t: torch.Tensor) -> torch.nn.Module:
        """
        Returns an instance of the Hamiltonian evolution object of the appropriate type.

        Args:
            H (tensor): The Hamiltonian to be used in the evolution.
            t (tensor): The evolution time.

        Returns:
            An instance of the Hamiltonian evolution object of the appropriate type.

        """

        return self.hamevo_type.value(H, t, self.qubits, self.n_qubits, self.n_steps)

    def forward(self, H: torch.Tensor, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Performs Hamiltonian evolution on the given state.

        Args:
            H (tensor): The Hamiltonian to be used in the evolution.
            t (tensor): The evolution time.
            state (tensor): The state on which to perform Hamiltonian evolution.

        Returns:
            The state (tensor) after Hamiltonian evolution.
        """
        ham_evo_instance = self.get_hamevo_instance(H, t)
        return ham_evo_instance.forward(state)
