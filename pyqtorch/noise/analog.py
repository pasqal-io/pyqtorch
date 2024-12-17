from __future__ import annotations
import torch
from torch import Tensor
from typing import Any, Callable
import math

from pyqtorch.time_dependent.mesolve import mesolve
from pyqtorch.utils import SolverType, DensityMatrix, density_mat
from pyqtorch.matrices import XMAT, YMAT, ZMAT

class AnalogNoise(torch.nn.Module):
    def __init__(
        self,
        noise_params: list[Tensor | float],
    ) -> None:
        if isinstance(noise_params[0], float):
            noise_params = [torch.tensor(p) for p in noise_params]
        for index, tensor in enumerate(noise_params):
            self.register_buffer(f"noise_operators_{index}", tensor)

        self._device: torch.device = noise_params[0].device
        self._dtype: torch.dtype = noise_params[0].dtype

    def extra_repr(self) -> str:
        return f"noise_params: {self.noise_params}"

    @property
    def noise_params(self) -> list[Tensor]:
        return [getattr(self, f"noise_params_{i}") for i in range(len(self._buffers))]

    def to(self, *args: Any, **kwargs: Any) -> AnalogNoise:
        super().to(*args, **kwargs)
        self._device = self.noise_params_0.device
        self._dtype = self.noise_params_0.dtype
        return self
    
    def forward(
        self,
        H: Callable[..., Any],
        state: Tensor,
        tsave: list | Tensor,
        solver: SolverType,
        options: dict[str, Any] | None = None,
    ) -> Tensor:
        if not isinstance(state, DensityMatrix):
            state = density_mat(state)
        sol = mesolve(
            H,
            state,
            self.noise_params,
            tsave,
            solver,
            options=options,
        )
        return DensityMatrix(sol.states[-1])
        

class Depolarizing(AnalogNoise):
    def __init__(self, error_param: float) -> None:
        coeff = math.sqrt(error_param / 4)
        L0 = coeff * XMAT.squeeze()
        L1 = coeff * YMAT.squeeze()
        L2 = coeff * ZMAT.squeeze()
        noise_operators: list[Tensor] = [L0, L1, L2]
        super().__init__(noise_operators)