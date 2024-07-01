from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor

from pyqtorch.time_dependent.options import AdaptiveSolverOptions
from pyqtorch.time_dependent.solvers import MEDormandPrince5
from pyqtorch.utils import Result, SolverType


def mesolve(
    H: Callable[..., Any],
    psi0: Tensor,
    L: list[Tensor],
    tsave: list | Tensor,
    solver: SolverType,
    options: dict[str, Any] = {},
) -> Result:
    """Solve time-dependent Lindblad master equation.

    Args:
        H (Callable[[float], Tensor]): time-dependent Hamiltonian of the system
        psi0 (Tensor): initial state or density matrix of the system
        L (list[Tensor]): list of jump operators
        tsave (Tensor): tensor containing simulation time instants
        solver (SolverType): name of the solver to use
        options (dict[str, Any], optional): additional options passed to the solver. Defaults to {}.

    Returns:
        Result: dataclass containing the simulated density matrices at each time moment
    """

    L = torch.stack(L)
    if psi0.size(-2) == 1:
        rho0 = psi0.mH @ psi0
    elif psi0.size(-1) == 1:
        rho0 = psi0 @ psi0.mH
    elif psi0.size(-1) == psi0.size(-2):
        rho0 = psi0
    else:
        raise ValueError(
            "Argument `psi0` must be a ket, bra or density matrix, but has shape"
            f" {tuple(psi0.shape)}."
        )

    # instantiate appropriate solver
    if solver == SolverType.DP5_ME:
        opt = AdaptiveSolverOptions(**options)
        s = MEDormandPrince5(H, rho0, L, tsave, opt)
    else:
        raise ValueError("Requested solver is not available.")

    # compute the result
    result = s.run()

    return Result(result)
