from __future__ import annotations

import random
from math import isclose

import numpy as np
import pytest
import torch

import pyqtorch.modules as pyq

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)


pi = torch.tensor(torch.pi, dtype=torch.cdouble)


def overlap(state1: torch.Tensor, state2: torch.Tensor) -> float | list[float]:
    N = len(state1.shape) - 1
    state1_T = torch.transpose(state1, N, 0)
    overlap = torch.tensordot(state1_T, state2, dims=N)
    res: list[float] = list(map(float, torch.abs(overlap**2).flatten()))
    if len(res) == 1:
        return res[0]
    else:
        return res


def Hamiltonian(batch_size: int = 1) -> torch.Tensor:
    sigmaz = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.cdouble))
    Hbase = torch.kron(sigmaz, sigmaz)
    H = torch.kron(Hbase, Hbase)
    if batch_size == 1:
        return H
    else:
        return torch.stack((H, H.conj()), dim=2)


@pytest.mark.parametrize(
    "ham_evo",
    [pyq.HamEvo, pyq.HamEvoEig],
)
def test_hamevo_modules_single(ham_evo: torch.nn.Module) -> None:
    n_qubits = 4
    H = Hamiltonian(1)
    t_evo = torch.tensor([torch.pi / 4], dtype=torch.cdouble)
    hamevo = ham_evo(H, t_evo, range(n_qubits), n_qubits)
    psi = pyq.uniform_state(n_qubits)
    psi_star = hamevo.forward(psi)
    result = overlap(psi_star, psi)
    result = result if isinstance(result, float) else result[0]
    assert isclose(result, 0.5)


@pytest.mark.parametrize(
    "ham_evo",
    [pyq.HamEvo, pyq.HamEvoEig],
)
def test_hamevo_modules_batch(ham_evo: torch.nn.Module) -> None:
    n_qubits = 4
    batch_size = 2
    H = Hamiltonian(batch_size)
    t_evo = torch.tensor([torch.pi / 4], dtype=torch.cdouble)

    hamevo = ham_evo(H, t_evo, range(n_qubits), n_qubits)
    psi = pyq.uniform_state(n_qubits, batch_size)
    psi_star = hamevo.forward(psi)
    result = overlap(psi_star, psi)
    print(result)

    assert map(isclose, zip(result, [0.5, 0.5]))  # type: ignore [arg-type]
