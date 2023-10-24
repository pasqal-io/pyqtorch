from __future__ import annotations

import torch

from pyqtorch.apply import _apply_einsum
from pyqtorch.utils import Operator, State, is_diag

BATCH_DIM = 2


class HamiltonianEvolution(torch.nn.Module):
    def __init__(
        self,
        hamiltonian: Operator,
        time_evolution: torch.Tensor,
        qubit_support: list[int],
        n_qubits: int = None,
    ):
        super().__init__()
        self.qubit_support: list[int] = qubit_support
        self.time_evolution: torch.Tensor = time_evolution
        self.hamiltonian: Operator = hamiltonian

        if n_qubits is None:
            n_qubits = len(qubit_support)
        self.n_qubits: int = n_qubits
        if len(self.hamiltonian.size()) < 3:
            self.hamiltonian = self.hamiltonian.unsqueeze(2)
        self.batch_size = max(self.hamiltonian.size()[2], len(self.time_evolution))
        diag_check = torch.tensor(
            [is_diag(self.hamiltonian[..., i]) for i in range(self.hamiltonian.size()[BATCH_DIM])]
        )
        batch_is_diag = bool(torch.prod(diag_check))

        def _diag_operator(hamiltonian: Operator, time_evolution: torch.Tensor) -> Operator:
            evol_operator = torch.diagonal(hamiltonian) * (-1j * time_evolution).view((-1, 1))
            evol_operator = torch.diag_embed(torch.exp(evol_operator))
            return torch.transpose(evol_operator, 0, -1)

        def _matrixexp_operator(hamiltonian: Operator, time_evolution: torch.Tensor) -> Operator:
            evol_operator = torch.transpose(hamiltonian, 0, -1) * (-1j * time_evolution).view(
                (-1, 1, 1)
            )
            evol_operator = torch.linalg.matrix_exp(evol_operator)
            return torch.transpose(evol_operator, 0, -1)

        self._evolve_operator = _diag_operator if batch_is_diag else _matrixexp_operator

    def forward(self, state: State) -> State:
        return _apply_einsum(
            state,
            self._evolve_operator(self.hamiltonian, self.time_evolution),
            self.qubit_support,
            self.n_qubits,
            self.batch_size,
        )
