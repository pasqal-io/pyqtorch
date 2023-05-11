from __future__ import annotations

import random
from math import isclose

import networkx as nx
import numpy as np
import torch

from pyqtorch.core import batched_operation, circuit, operation
from pyqtorch.core.batched_operation import (
    batched_hamiltonian_evolution,
    batched_hamiltonian_evolution_eig,
)
from pyqtorch.core.circuit import QuantumCircuit
from pyqtorch.core.operation import hamiltonian_evolution, hamiltonian_evolution_eig
from pyqtorch.matrices import generate_ising_from_graph

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(not torch.cuda.is_available())

state_00 = torch.tensor([[1, 0], [0, 0]], dtype=torch.cdouble).unsqueeze(2)
state_10 = torch.tensor([[0, 1], [0, 0]], dtype=torch.cdouble).unsqueeze(2)
state_01 = torch.tensor([[0, 0], [1, 0]], dtype=torch.cdouble).unsqueeze(2)
state_11 = torch.tensor([[0, 0], [0, 1]], dtype=torch.cdouble).unsqueeze(2)

pi = torch.tensor(torch.pi, dtype=torch.cdouble)

CNOT_mat: torch.Tensor = torch.tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.cdouble
)


def test_hamevo_single() -> None:
    N = 4
    qc = circuit.QuantumCircuit(N)
    psi = qc.uniform_state(1)

    def overlap(state1: torch.Tensor, state2: torch.Tensor) -> float:
        N = len(state1.shape) - 1
        state1_T = torch.transpose(state1, N, 0)
        overlap = torch.tensordot(state1_T, state2, dims=N)
        return float(torch.abs(overlap**2).flatten())

    sigmaz = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.cdouble))
    Hbase = torch.kron(sigmaz, sigmaz)
    H = torch.kron(Hbase, Hbase)
    t_evo = torch.tensor([torch.pi / 4], dtype=torch.cdouble)
    psi_star = operation.hamiltonian_evolution(H, psi, t_evo, range(N), N)
    result: float = overlap(psi_star, psi)

    assert isclose(result, 0.5)


def test_hamevo_eig_single() -> None:
    N = 4
    qc = circuit.QuantumCircuit(N)
    psi = qc.uniform_state(1)

    def overlap(state1: torch.Tensor, state2: torch.Tensor) -> float:
        N = len(state1.shape) - 1
        state1_T = torch.transpose(state1, N, 0)
        overlap = torch.tensordot(state1_T, state2, dims=N)
        return float(torch.abs(overlap**2).flatten())

    sigmaz = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.cdouble))
    Hbase = torch.kron(sigmaz, sigmaz)
    H = torch.kron(Hbase, Hbase)
    t_evo = torch.tensor([torch.pi / 4], dtype=torch.cdouble)
    psi_star = operation.hamiltonian_evolution_eig(H, psi, t_evo, range(N), N)
    result: float = overlap(psi_star, psi)

    assert isclose(result, 0.5)


def test_hamevo_batch() -> None:
    N = 4
    qc = circuit.QuantumCircuit(N)
    psi = qc.uniform_state(batch_size=2)

    def overlap(state1: torch.Tensor, state2: torch.Tensor) -> list[float]:
        N = len(state1.shape) - 1
        state1_T = torch.transpose(state1, N, 0)
        overlap = torch.tensordot(state1_T, state2, dims=N)
        return list(map(float, torch.abs(overlap**2).flatten()))

    sigmaz = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.cdouble))
    Hbase = torch.kron(sigmaz, sigmaz)
    H = torch.kron(Hbase, Hbase)
    H_conj = H.conj()

    t_evo = torch.tensor([0], dtype=torch.cdouble)
    psi = operation.hamiltonian_evolution(H, psi, t_evo, range(N), N)

    t_evo = torch.tensor([torch.pi / 4], dtype=torch.cdouble)
    psi_star = operation.hamiltonian_evolution(H, psi, t_evo, range(N), N)
    H_batch = torch.stack((H, H_conj), dim=2)
    batched_operation.batched_hamiltonian_evolution(H_batch, psi, t_evo, range(N), N)
    result: list[float] = overlap(psi_star, psi)

    assert map(isclose, zip(result, [0.5, 0.5]))  # type: ignore [arg-type]


def test_hamevo_rk4_vs_eig_diag_H() -> None:
    n_qubits: int = 7
    batch_size: int = 10
    graph: nx.Graph = nx.fast_gnp_random_graph(n_qubits, 0.7)
    qc = QuantumCircuit(n_qubits)
    psi = qc.uniform_state(batch_size)

    H_diag = generate_ising_from_graph(graph)
    H = torch.diag(H_diag)

    n_trials = 10
    wf_save_rk = torch.zeros((n_trials,) + tuple(psi.shape)).to(torch.cdouble)
    wf_save_eig = torch.zeros((n_trials,) + tuple(psi.shape)).to(torch.cdouble)

    for i in range(n_trials):
        t_evo = torch.rand(batch_size) * 0.5

        psi_star = hamiltonian_evolution(H, psi, t_evo, range(n_qubits), n_qubits)
        wf_save_rk[i] = psi_star

        psi_star = hamiltonian_evolution_eig(H, psi, t_evo, range(n_qubits), n_qubits)
        wf_save_eig[i] = psi_star

    diff = torch.tensor(
        [torch.max(abs(wf_save_rk[i, ...] - wf_save_eig[i, ...])) for i in range(n_trials)]
    )

    assert torch.max(diff) <= 10 ** (-6)


def test_hamevo_rk4_vs_eig_general_H() -> None:
    n_qubits: int = 6
    batch_size: int = 10

    qc = QuantumCircuit(n_qubits)
    psi = qc.uniform_state(batch_size)

    H_0 = torch.randn((2**n_qubits, 2**n_qubits), dtype=torch.cdouble)
    H = (H_0 + torch.conj(H_0.transpose(0, 1))).to(torch.cdouble)

    n_trials = 10
    wf_save_rk = torch.zeros((n_trials,) + tuple(psi.shape)).to(torch.cdouble)
    wf_save_eig = torch.zeros((n_trials,) + tuple(psi.shape)).to(torch.cdouble)

    for i in range(n_trials):
        t_evo = torch.rand(batch_size) * 0.5
        psi_star = hamiltonian_evolution(H, psi, t_evo, range(n_qubits), n_qubits)
        wf_save_rk[i] = psi_star
        psi_star = hamiltonian_evolution_eig(H, psi, t_evo, range(n_qubits), n_qubits)
        wf_save_eig[i] = psi_star

    diff = torch.tensor(
        [torch.max(abs(wf_save_rk[i, ...] - wf_save_eig[i, ...])) for i in range(n_trials)]
    )

    assert torch.max(diff) <= 10 ** (-5)


def test_hamevo_rk4_vs_eig_general_H_batched() -> None:
    n_qubits: int = 5
    batch_size: int = 20

    qc = QuantumCircuit(n_qubits)
    psi = qc.uniform_state(batch_size)

    H_batch = torch.zeros(2**n_qubits, 2**n_qubits, batch_size).to(torch.cdouble)
    for i in range(batch_size):
        H_0 = torch.randn((2**n_qubits, 2**n_qubits), dtype=torch.cdouble)
        H_batch[..., i] = (H_0 + torch.conj(H_0.transpose(0, 1))).to(torch.cdouble)

    t_evo = (torch.rand(batch_size) * 0.5).to(torch.cdouble)

    psi_star_norm = batched_hamiltonian_evolution(H_batch, psi, t_evo, range(n_qubits), n_qubits)

    psi_star_eig = batched_hamiltonian_evolution_eig(H_batch, psi, t_evo, range(n_qubits), n_qubits)

    diff = torch.tensor(
        [torch.max(abs(psi_star_norm[..., b] - psi_star_eig[..., b])) for b in range(batch_size)]
    )

    assert torch.max(diff) <= 10 ** (-6)
