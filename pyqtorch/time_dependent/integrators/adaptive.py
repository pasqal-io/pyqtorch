from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

import torch
from torch import Tensor

from pyqtorch.time_dependent.options import AdaptiveSolverOptions
from pyqtorch.utils import hairer_norm


class AdaptiveIntegrator:
    """Adaptive step-size ODE integrator.

    For details about the integration method, see Chapter II.4 of [1].

    [1] Hairer et al., Solving Ordinary Differential Equations I (1993), Springer
        Series in Computational Mathematics.
    """

    def __init__(
        self,
        H: Tensor | Callable,
        y0: Tensor,
        tsave: Tensor,
        options: AdaptiveSolverOptions,
    ):
        self.H = H
        self.t0 = 0.0
        self.y0 = y0
        self.tsave = tsave
        self.options = options

        # initialize the step counter
        self.step_counter = 0

    def init_forward(self) -> tuple:
        # initial values of the ODE routine
        f0 = self.ode_fun(self.t0, self.y0)
        dt0 = self.init_tstep(self.t0, self.y0, f0, self.ode_fun)
        error0 = 1.0
        return self.t0, self.y0, f0, dt0, error0

    @property
    @abstractmethod
    def order(self) -> int:
        pass

    @property
    @abstractmethod
    def tableau(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pass

    @abstractmethod
    def step(
        self,
        t0: float,
        y0: Tensor,
        f0: Tensor,
        dt: float,
        fun: Callable[[float, Tensor], Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute a single step of the ODE integration."""
        pass

    @abstractmethod
    def ode_fun(self, t: float, y: Tensor) -> Tensor:
        pass

    def integrate(self, t0: float, t1: float, y: Tensor, *args: Any) -> tuple:
        ft, dt, error = args

        cache = (dt, error)
        t = t0
        while t < t1:

            dt = self.update_tstep(dt, error)

            # check for time overflow
            if t + dt >= t1:
                cache = (dt, error)
                dt = t1 - t

            # compute the next step
            ft_new, y_new, y_err = self.step(t, y, ft, dt, self.ode_fun)

            error = self.get_error(y_err, y, y_new)
            if error <= 1:
                t, y, ft = t + dt, y_new, ft_new

            # check max steps are not reached
            self.increment_step_counter(t)

        dt, error = cache
        return y, ft, dt, error

    def increment_step_counter(self, t: float) -> None:
        """Increment the step counter and check for max steps."""
        self.step_counter += 1
        if self.step_counter == self.options.max_steps:
            raise RuntimeError(
                "Maximum number of time steps reached in adaptive time step ODE"
                f" solver at time t={t:.2g} (`max_steps={self.options.max_steps}`)."
                " This is likely due to a diverging solution. Try increasing the"
                " maximum number of steps, or use a different solver."
            )

    @torch.no_grad()
    def get_error(self, y_err: Tensor, y0: Tensor, y1: Tensor) -> float:
        """Compute the error of a given solution.

        See Equation (4.11) of [1].
        """
        scale = self.options.atol + self.options.rtol * torch.max(y0.abs(), y1.abs())
        return float(hairer_norm(y_err / scale).max().item())

    @torch.no_grad()
    def init_tstep(
        self, t0: float, y0: Tensor, f0: Tensor, fun: Callable[[float, Tensor], Tensor]
    ) -> float:
        """Initialize the time step of an adaptive step size integrator.

        See Equation (4.14) of [1] for the detailed steps. For this function, we keep
        the same notations as in the book.
        """

        sc = self.options.atol + torch.abs(y0) * self.options.rtol
        f0 = f0.to_dense() if self.options.use_sparse else f0
        d0, d1 = (
            hairer_norm(y0 / sc).max().item(),
            hairer_norm(f0 / sc).max().item(),
        )

        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1

        y1 = y0 + h0 * f0
        f1 = fun(t0 + h0, y1)
        diff = (f1 - f0).to_dense() if self.options.use_sparse else f1 - f0
        d2 = hairer_norm(diff / sc).max().item() / h0
        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** (1.0 / float(self.order + 1))

        return min(100 * h0, h1)

    @torch.no_grad()
    def update_tstep(self, dt: float, error: float) -> float:
        """Update the time step of an adaptive step size integrator.

        See Equation (4.12) and (4.13) of [1] for the detailed steps.
        """
        if error == 0:  # no error -> maximally increase the time step
            return dt * self.options.max_factor

        elif error <= 1:  # time step accepted -> take next time step at least as large
            return float(
                dt
                * max(
                    1.0,
                    min(
                        self.options.max_factor,
                        self.options.safety_factor * error ** (-1.0 / self.order),
                    ),
                )
            )

        else:  # time step rejected -> reduce next time step
            return float(
                dt
                * max(
                    self.options.min_factor,
                    self.options.safety_factor * error ** (-1.0 / self.order),
                )
            )

    def run(self) -> Tensor:
        """Integrates the ODE forward from time `self.t0` to time `self.tstop[-1]`
        starting from initial state `self.y0`, and save the state for each time in
        `self.tstop`."""

        # initialize the ODE routine
        t, y, *args = self.init_forward()

        # run the ODE routine
        result = []
        for tnext in self.tsave:
            y, *args = self.integrate(t, tnext, y, *args)
            result.append(y)
            t = tnext

        if len(y.shape) == 2:
            res = torch.stack(result)
        elif len(y.shape) == 3:
            res = torch.stack(result).permute(0, 2, 3, 1)

        return res
