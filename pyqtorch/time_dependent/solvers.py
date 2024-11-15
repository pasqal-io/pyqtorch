from __future__ import annotations

import warnings
from typing import Any, Callable

import torch
from torch import Tensor

from pyqtorch.time_dependent.integrators.adaptive import AdaptiveIntegrator
from pyqtorch.time_dependent.methods.dp5 import DormandPrince5
from pyqtorch.time_dependent.methods.krylov import Krylov
from pyqtorch.time_dependent.options import AdaptiveSolverOptions
from pyqtorch.utils import cache


class SESolver(AdaptiveIntegrator):
    """Generic adaptive-step solver for Schrodinger equation."""

    def __init__(
        self,
        H: Tensor | Callable[..., Any],
        psi0: Tensor,
        tsave: Tensor,
        options: AdaptiveSolverOptions,
    ):
        super().__init__(H, psi0, tsave, options)

    def ode_fun(self, t: float, psi: Tensor) -> Tensor:
        """Compute dpsi / dt = -1j * H(psi) at time t."""
        with warnings.catch_warnings():
            # filter-out UserWarning about "Sparse CSR tensor support is in beta state"
            warnings.filterwarnings("ignore", category=UserWarning)
            res = (
                -1j * self.H(t) @ (psi.to_sparse() if self.options.use_sparse else psi)
            )
        return res


class MESolver(AdaptiveIntegrator):
    """Generic adaptive-step solver for Linblad master equation."""

    def __init__(
        self,
        H: Callable[..., Any],
        rho0: Tensor,
        L: Tensor,
        tsave: Tensor,
        options: AdaptiveSolverOptions,
    ):
        super().__init__(H, rho0, tsave, options)

        self.L = L
        self.L_tuple = torch.unbind(self.L)

        L_concat_v = torch.cat(self.L_tuple)

        self.sum_LdagL = L_concat_v.mH @ L_concat_v

        # define identity operator
        n = self.H(0.0).size(-1)
        self.I = torch.eye(n, dtype=options.ctype)

        # define cached non-hermitian Hamiltonian
        self.Hnh = cache(lambda H: H - 0.5j * self.sum_LdagL)

    def ode_fun(self, t: float, rho: Tensor) -> Tensor:
        H = self.H(t)
        lindblad_term = sum(L @ rho @ L.mH for L in self.L_tuple)
        out: Tensor = -1j * self.Hnh(H) @ rho + 0.5 * lindblad_term
        return out + out.mH


class SEDormandPrince5(SESolver, DormandPrince5):
    pass


class MEDormandPrince5(MESolver, DormandPrince5):
    pass


class SEKrylov(Krylov):
    pass
