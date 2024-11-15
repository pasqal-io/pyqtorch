from __future__ import annotations

from abc import abstractmethod
from typing import Callable

import torch
from torch import Tensor

from pyqtorch.time_dependent.options import KrylovSolverOptions


class KrylovIntegrator:
    """Krylov subspace method to evolve the state vector."""

    def __init__(
        self,
        H: Callable,
        y0: Tensor,
        tsave: Tensor,
        options: KrylovSolverOptions,
    ):
        self.H = H
        self.t0 = 0.0
        self.y0 = y0
        self.tsave = tsave
        self.options = options

    @abstractmethod
    def integrate(self, t0: float, t1: float, y: Tensor) -> Tensor:
        pass

    def run(self) -> Tensor:

        out = []
        for i in range(self.y0.shape[1]):
            # run the Krylov routine
            result = []
            y = self.y0[:, i : i + 1]
            t = self.t0
            for tnext in self.tsave:
                y = self.integrate(t, tnext, y)
                result.append(y.T)
                t = tnext
            out.append(torch.cat(result))
        return torch.stack(out, dim=2)
