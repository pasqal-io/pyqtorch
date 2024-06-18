from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Callable

import torch
from torch import Tensor
from tqdm import TqdmWarning, tqdm

from pyqtorch.time_dependent.options import KrylovSolverOptions


class KrylovIntegrator:
    """Uses Krylov subspace method to evolve the state vector."""

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

        # initialize the progress bar
        self.pbar = tqdm(total=float(self.tsave[-1]), disable=not self.options.verbose)

    @abstractmethod
    def integrate(self, t0: float, t1: float, y: Tensor) -> Tensor:
        pass

    def run(self) -> Tensor:
        t = self.t0

        # run the Krylov routine
        result = []
        y = self.y0
        for tnext in self.tsave:
            y = self.integrate(t, tnext, y)
            result.append(y.T)
            t = tnext

        # close the progress bar
        with warnings.catch_warnings():  # ignore tqdm precision overflow
            warnings.simplefilter("ignore", TqdmWarning)
            self.pbar.close()

        return torch.cat(result).unsqueeze(-1)
