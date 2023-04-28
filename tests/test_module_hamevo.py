from __future__ import annotations

import copy
import random
from math import isclose

import numpy as np
import torch

import pyqtorch.modules as pyq

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)


pi = torch.tensor(torch.pi, dtype=torch.cdouble)


def test_hamevo_module_single() -> None:
    n_qubits = 4
    sigmaz = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.cdouble))
    Hbase = torch.kron(sigmaz, sigmaz)
    H = torch.kron(Hbase, Hbase)
    print(H.shape)

    # H = H.unsqueeze(2)
    print(H.shape)

    t_evo = torch.tensor([torch.pi / 4], dtype=torch.cdouble)
    # t_evo = t_evo.unsqueeze(1)
    hamevo = pyq.HamEvo(H, t_evo, range(n_qubits), n_qubits)
    psi = pyq.uniform_state(n_qubits)

    psi_0 = copy.deepcopy(psi)

    def overlap(state1: torch.Tensor, state2: torch.Tensor) -> float:
        N = len(state1.shape) - 1
        state1_T = torch.transpose(state1, N, 0)
        overlap = torch.tensordot(state1_T, state2, dims=N)
        return float(torch.abs(overlap**2).flatten())

    psi_star = hamevo.forward(psi)
    # result = overlap(psi, psi_star)
    result: float = overlap(psi_star, psi_0)

    assert isclose(result, 0.5)
