from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import dtype


@dataclass
class AdaptiveSolverOptions:

    atol: float = 1e-8
    rtol: float = 1e-6
    max_steps: int = 100_000
    safety_factor: float = 0.9
    min_factor: float = 0.2
    max_factor: float = 5.0
    ctype: dtype = torch.complex128
    rtype: dtype = torch.float64
    use_sparse: bool = False


@dataclass
class KrylovSolverOptions:

    max_krylov: int = 80
    exp_tolerance: float = 1e-10
    norm_tolerance: float = 1e-10
    use_sparse: bool = False
