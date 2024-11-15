from __future__ import annotations

import functools
from typing import Any, Callable

import torch
from torch import Tensor

from pyqtorch.time_dependent.integrators.adaptive import AdaptiveIntegrator


class DormandPrince5(AdaptiveIntegrator):
    """Dormand-Prince method for adaptive time step ODE integration.

    This is a fifth order solver that uses a fourth order solution to estimate the
    integration error. It does so using only six function evaluations. See `Dormand and
    Prince, A family of embedded Runge-Kutta formulae (1980), Journal of Computational
    and Applied Mathematics`. See also `Shampine, Some Practical Runge-Kutta Formulas
    (1986), Mathematics of Computation`.
    """

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)

    @functools.cached_property
    def order(self) -> int:
        return 5

    @functools.cached_property
    def tableau(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Build the Butcher tableau of the integrator."""
        alpha = torch.tensor(
            [1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0, 0.0], dtype=self.options.rtype
        )
        beta = torch.tensor(
            [
                [1 / 5, 0, 0, 0, 0, 0, 0],
                [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
                [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
                [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
                [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
                [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
            ],
            dtype=self.options.ctype,
        )
        csol5 = torch.tensor(
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
            dtype=self.options.ctype,
        )
        csol4 = torch.tensor(
            [
                5179 / 57600,
                0,
                7571 / 16695,
                393 / 640,
                -92097 / 339200,
                187 / 2100,
                1 / 40,
            ],
            dtype=self.options.ctype,
        )

        return alpha, beta, csol5, csol5 - csol4

    def step(
        self,
        t: float,
        y: Tensor,
        f: Tensor,
        dt: float,
        fun: Callable[[float, Tensor], Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        # import butcher tableau
        alpha, beta, csol, cerr = self.tableau

        # compute iterated Runge-Kutta values
        k = torch.empty(7, *f.shape, dtype=self.options.ctype)
        k[0] = f.to_dense() if self.options.use_sparse else f
        for i in range(1, 7):
            dy = torch.tensordot(dt * beta[i - 1, :i], k[:i].clone(), dims=([0], [0]))
            a = fun(t + dt * alpha[i - 1].item(), y + dy)
            k[i] = a.to_dense() if self.options.use_sparse else a

        # compute results
        f_new = k[-1]
        y_new = y + torch.tensordot(dt * csol[:6], k[:6], dims=([0], [0]))
        y_err = torch.tensordot(dt * cerr, k, dims=([0], [0]))

        return f_new, y_new, y_err
