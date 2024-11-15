from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import qutip
import torch
from qutip import Qobj
from torch import Tensor

from pyqtorch.time_dependent.mesolve import mesolve
from pyqtorch.time_dependent.sesolve import sesolve
from pyqtorch.utils import SolverType

ATOL = 5e-2


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("ode_solver", [SolverType.DP5_SE, SolverType.KRYLOV_SE])
def test_sesolve(
    duration: float,
    batch_size: int,
    n_steps: int,
    torch_hamiltonian: Callable,
    qutip_hamiltonian: Callable,
    ode_solver: SolverType,
) -> None:
    psi0_qutip = qutip.basis(4, 0)

    # simulate with torch-based solver
    psi0_torch = (
        torch.tensor(psi0_qutip.full()).to(torch.complex128).repeat(1, batch_size)
    )

    t_points = torch.linspace(0, duration, n_steps)
    state_torch = sesolve(torch_hamiltonian, psi0_torch, t_points, ode_solver).states[
        -1
    ]

    # simulate with qutip solver
    t_points = np.linspace(0, duration, n_steps)
    result = qutip.sesolve(qutip_hamiltonian, psi0_qutip, t_points)
    state_qutip = torch.tensor(result.states[-1].full()).repeat(1, batch_size)

    assert torch.allclose(state_torch, state_qutip, atol=ATOL)


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("ode_solver", [SolverType.DP5_ME])
def test_mesolve(
    duration: float,
    batch_size: int,
    n_steps: int,
    torch_hamiltonian: Callable,
    qutip_hamiltonian: Callable,
    jump_op_torch: list[Tensor],
    jump_op_qutip: list[Qobj],
    ode_solver: SolverType,
) -> None:
    psi0_qutip = qutip.basis(4, 0)

    # simulate with torch-based solver
    psi0_torch = torch.tensor(psi0_qutip.full()).to(torch.complex128)
    rho0_torch = (
        torch.matmul(psi0_torch, psi0_torch.T).unsqueeze(-1).repeat(1, 1, batch_size)
    )

    t_points = torch.linspace(0, duration, n_steps)
    state_torch = mesolve(
        torch_hamiltonian, rho0_torch, jump_op_torch, t_points, ode_solver
    ).states[-1]

    # simulate with qutip solver
    t_points = np.linspace(0, duration, n_steps)
    result = qutip.mesolve(qutip_hamiltonian, psi0_qutip, t_points, jump_op_qutip)
    state_qutip = (
        torch.tensor(result.states[-1].full()).unsqueeze(-1).repeat(1, 1, batch_size)
    )

    assert torch.allclose(state_torch, state_qutip, atol=ATOL)
