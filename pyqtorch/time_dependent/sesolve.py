from __future__ import annotations

from typing import Any, Callable

from torch import Tensor

from pyqtorch.time_dependent.options import AdaptiveSolverOptions, KrylovSolverOptions
from pyqtorch.time_dependent.solvers import SEDormandPrince5, SEKrylov
from pyqtorch.utils import Result, SolverType


def sesolve(
    H: Callable[[float], Tensor],
    psi0: Tensor,
    tsave: Tensor,
    solver: SolverType,
    options: dict[str, Any] | None = None,
) -> Result:
    """Solve time-dependent Schrodinger equation.

    Args:
        H (Callable[[float], Tensor]): time-dependent Hamiltonian of the system
        psi0 (Tensor): initial state of the system
        tsave (Tensor): tensor containing simulation time instants
        solver (SolverType): name of the solver to use
        options (dict[str, Any], optional): additional options passed to the solver.
            Defaults to None.

    Returns:
        Result: dataclass containing the simulated states at each time moment
    """
    # check dimensions of initial state
    n = H(0.0).shape[0]
    if (psi0.shape[0] != n) or len(psi0.shape) != 2:
        raise ValueError(
            f"Argument `psi0` must be a 2D tensor of shape `({n}, batch_size)`. Current shape:"
            f" {tuple(psi0.shape)}."
        )

    options = options or dict()
    # instantiate appropriate solver
    if solver == SolverType.DP5_SE:
        opt = AdaptiveSolverOptions(**options)
        s = SEDormandPrince5(H, psi0, tsave, opt)
    elif solver == SolverType.KRYLOV_SE:
        opt = KrylovSolverOptions(**options)  # type: ignore [assignment]
        s = SEKrylov(H, psi0, tsave, opt)  # type: ignore [assignment, arg-type]
    else:
        raise ValueError("Requested solver is not available.")

    # compute the result
    result = s.run()

    return Result(result)
