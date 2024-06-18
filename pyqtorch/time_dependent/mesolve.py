from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor

from pyqtorch.time_dependent.options import AdaptiveSolverOptions
from pyqtorch.time_dependent.solvers import MEDormandPrince5
from pyqtorch.utils import Result, SolverType, density_mat


def mesolve(
    H: Callable[..., Any],
    rho0: Tensor,
    L: list[Tensor],
    tsave: list | Tensor,
    solver: SolverType,
    options: dict[str, Any] = {},
) -> Result:
    """Solve time-dependent Lindblad master equation.

    Args:
        H (Callable[[float], Tensor]): time-dependent Hamiltonian of the system
        rho0 (Tensor): initial state or density matrix of the system
        L (list[Tensor]): list of jump operators
        tsave (Tensor): tensor containing simulation time instants
        solver (SolverType): name of the solver to use
        options (dict[str, Any], optional): additional options passed to the solver. Defaults to {}.

    Returns:
        Result: dataclass containing the simulated density matrices at each time moment
    """

    L = torch.stack(L)
    if rho0.shape[1] == 1:
        rho0 = density_mat(rho0)

    # instantiate appropriate solver
    if solver == SolverType.DP5_ME:
        opt = AdaptiveSolverOptions(**options)
        s = MEDormandPrince5(H, rho0, L, tsave, opt)
    else:
        raise ValueError("Requested solver is not available.")

    # compute the result
    result = s.run()

    return Result(result)
