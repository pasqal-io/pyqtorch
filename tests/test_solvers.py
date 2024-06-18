from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import qutip
import torch

from pyqtorch.time_dependent.sesolve import sesolve
from pyqtorch.utils import SolverType

ATOL = 5e-2


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("ode_solver", [SolverType.DP5_SE, SolverType.KRYLOV_SE])
def test_sesolve(
    torch_hamiltonian: Callable,
    qutip_hamiltonian: Callable,
    ode_solver: SolverType,
) -> None:
    duration = 1.0
    n_steps = 1000
    psi0_qutip = qutip.basis(4, 0)

    # simulate with torch-based ODE solver
    psi0_torch = torch.tensor(psi0_qutip.full()).to(torch.complex128)
    t_points = torch.linspace(0, duration, n_steps)
    state_torch = sesolve(torch_hamiltonian, psi0_torch, t_points, ode_solver).states[-1]

    # simulate with qutip
    t_points = np.linspace(0, duration, n_steps)
    result = qutip.sesolve(qutip_hamiltonian, psi0_qutip, t_points)
    state_qutip = torch.tensor(result.states[-1].full())

    assert torch.allclose(state_torch, state_qutip, atol=ATOL)
