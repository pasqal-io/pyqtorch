from __future__ import annotations

import random
from math import isclose
from typing import Callable

import numpy as np
import pytest
import torch

import pyqtorch.modules as pyq

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(not torch.cuda.is_available())


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


def Hamiltonian_general(n_qubits: int = 2, batch_size: int = 1) -> torch.Tensor:
    H_batch = torch.zeros((2**n_qubits, 2**n_qubits, batch_size), dtype=torch.cdouble)
    for i in range(batch_size):
        H_0 = torch.randn((2**n_qubits, 2**n_qubits), dtype=torch.cdouble)
        H = (H_0 + torch.conj(H_0.transpose(0, 1))).cdouble()
        H_batch[..., i] = H
    return H_batch


def Hamiltonian_diag(n_qubits: int = 2, batch_size: int = 1) -> torch.Tensor:
    H_batch = torch.zeros((2**n_qubits, 2**n_qubits, batch_size), dtype=torch.cdouble)
    for i in range(batch_size):
        H_0 = torch.randn((2**n_qubits, 2**n_qubits), dtype=torch.cdouble)
        H = (H_0 + torch.conj(H_0.transpose(0, 1))).cdouble()
        get_diag = torch.diag(H)
        H_diag = torch.diag(get_diag)
        H_batch[..., i] = H_diag
    return H_batch


@pytest.mark.parametrize(
    "ham_evo",
    [pyq.HamiltonianEvolution],
)
def test_ham_modules_single(ham_evo: torch.nn.Module) -> None:
    n_qubits = 4
    H = Hamiltonian(1)
    t_evo = torch.tensor([torch.pi / 4], dtype=torch.cdouble)
    hamevo = ham_evo(range(n_qubits), n_qubits)
    psi = pyq.uniform_state(n_qubits)
    psi_star = hamevo(H, t_evo, psi)
    result = overlap(psi_star, psi)
    result = result if isinstance(result, float) else result[0]
    assert isclose(result, 0.5)


@pytest.mark.parametrize(
    "ham_evo",
    [pyq.HamiltonianEvolution],
)
def test_hamiltonianevolution_batch(ham_evo: torch.nn.Module) -> None:
    n_qubits = 4
    batch_size = 2
    H = Hamiltonian(batch_size)
    t_evo = torch.tensor([torch.pi / 4], dtype=torch.cdouble)

    hamevo = ham_evo(range(n_qubits), n_qubits)
    psi = pyq.uniform_state(n_qubits, batch_size)
    psi_star = hamevo(H, t_evo, psi)
    result = overlap(psi_star, psi)

    assert map(isclose, zip(result, [0.5, 0.5]))  # type: ignore [arg-type]


@pytest.mark.parametrize(
    "ham_evo",
    [pyq.HamEvo, pyq.HamEvoEig, pyq.HamEvoExp],
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
    [pyq.HamEvo, pyq.HamEvoEig, pyq.HamEvoExp],
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


@pytest.mark.parametrize("get_hamiltonians", [Hamiltonian_general, Hamiltonian_diag])
def test_hamevo_consistency(get_hamiltonians: Callable) -> None:
    n_qubits = 4
    batch_size = 5

    H_batch = get_hamiltonians(n_qubits, batch_size)

    t_evo = torch.tensor([torch.pi / 8], dtype=torch.cdouble)
    psi_0 = pyq.uniform_state(batch_size=batch_size, n_qubits=n_qubits)

    hamevo_rk4 = pyq.HamEvo(H_batch, t_evo, range(n_qubits), n_qubits)
    psi_rk4 = hamevo_rk4.forward(psi_0)
    hamevo_eig = pyq.HamEvoEig(H_batch, t_evo, range(n_qubits), n_qubits)
    psi_eig = hamevo_eig.forward(psi_0)
    hamevo_exp = pyq.HamEvoExp(H_batch, t_evo, range(n_qubits), n_qubits)
    psi_exp = hamevo_exp.forward(psi_0)

    hamiltonian_evolution = pyq.HamiltonianEvolution(range(n_qubits), n_qubits)
    psi_ham = hamiltonian_evolution(H_batch, t_evo, psi_0)

    # assert torch.allclose(psi_rk4, psi_eig)
    # assert torch.allclose(psi_rk4, psi_eig)
    # assert torch.allclose(psi_eig, psi_exp)
    tensors = [psi_rk4, psi_eig, psi_exp, psi_ham]
    assert all(torch.allclose(tensors[i], tensors[0]) for i in range(1, len(tensors)))
