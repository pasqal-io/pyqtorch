from __future__ import annotations

import logging
from logging import getLogger
from typing import Tuple

import torch

from pyqtorch.apply import apply_operator
from pyqtorch.utils import Operator, State, is_diag

BATCH_DIM = 2


logger = getLogger(__name__)


def forward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Forward complete")
    torch.cuda.nvtx.range_pop()


def pre_forward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Executing forward")
    torch.cuda.nvtx.range_push("HamiltonianEvolution.forward")


def backward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Backward complete")
    torch.cuda.nvtx.range_pop()


def pre_backward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Executed backward")
    torch.cuda.nvtx.range_push("Hamiltonian Evolution.backward")


class HamiltonianEvolution(torch.nn.Module):
    def __init__(
        self,
        qubit_support: Tuple[int, ...],
        n_qubits: int = None,
    ):
        super().__init__()
        self.qubit_support: Tuple[int, ...] = qubit_support
        if n_qubits is None:
            n_qubits = len(qubit_support)
        self.n_qubits: int = n_qubits

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

        self._evolve_diag_operator = _diag_operator
        self._evolve_matrixexp_operator = _matrixexp_operator
        logger.debug("Hamiltonian Evolution initialized")
        if logger.isEnabledFor(logging.DEBUG):
            # When Debugging let's add logging and NVTX markers
            # WARNING: incurs performance penalty
            self.register_forward_hook(forward_hook, always_call=True)
            self.register_full_backward_hook(backward_hook)
            self.register_forward_pre_hook(pre_forward_hook)
            self.register_full_backward_pre_hook(pre_backward_hook)

    def forward(
        self,
        hamiltonian: Operator,
        time_evolution: torch.Tensor,
        state: State,
    ) -> State:
        if len(hamiltonian.size()) < 3:
            hamiltonian = hamiltonian.unsqueeze(2)
        self.batch_size = max(hamiltonian.size()[2], len(time_evolution))
        diag_check = torch.tensor(
            [is_diag(hamiltonian[..., i]) for i in range(hamiltonian.size()[BATCH_DIM])]
        )
        evolve_operator = (
            self._evolve_diag_operator
            if bool(torch.prod(diag_check))
            else self._evolve_matrixexp_operator
        )
        return apply_operator(
            state,
            evolve_operator(hamiltonian, time_evolution),
            self.qubit_support,
            self.n_qubits,
            self.batch_size,
        )
