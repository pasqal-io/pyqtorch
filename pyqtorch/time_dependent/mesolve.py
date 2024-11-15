from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor

from pyqtorch.time_dependent.options import AdaptiveSolverOptions
from pyqtorch.time_dependent.solvers import MEDormandPrince5
from pyqtorch.utils import Result, SolverType


def mesolve(
    H: Callable[..., Any],
    rho0: Tensor,
    L: list[Tensor],
    tsave: list | Tensor,
    solver: SolverType,
    options: dict[str, Any] | None = None,
) -> Result:
    """Solve time-dependent Lindblad master equation.

    Args:
        H (Callable[[float], Tensor]): time-dependent Hamiltonian of the system
        rho0 (Tensor): initial density matrix of the system
        L (list[Tensor]): list of jump operators
        tsave (Tensor): tensor containing simulation time instants
        solver (SolverType): name of the solver to use
        options (dict[str, Any], optional): additional options passed to the solver.
            Defaults to None.

    Returns:
        Result: dataclass containing the simulated density matrices at each time moment
    """
    options = options or dict()
    L = torch.stack(L)

    # check dimensions of initial state
    n = H(0.0).shape[0]
    if (
        (rho0.shape[0] != rho0.shape[1])
        or (rho0.shape[0] != n)
        or (len(rho0.shape) != 3)
    ):
        raise ValueError(
            f"Argument `rho0` must be a 3D tensor of shape `({n}, {n}, batch_size)`. "
            f"Current shape: {tuple(rho0.shape)}."
        )

    # permute dimensions to allow batch operations
    rho0 = rho0.permute(2, 0, 1)

    # instantiate appropriate solver
    if solver == SolverType.DP5_ME:
        opt = AdaptiveSolverOptions(**options)
        s = MEDormandPrince5(H, rho0, L, tsave, opt)
    else:
        raise ValueError("Requested solver is not available.")

    # compute the result
    result = s.run()

    return Result(result)
