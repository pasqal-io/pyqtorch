from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import torch
from conftest import _calc_mat_vec_wavefunction

import pyqtorch as pyq
from pyqtorch.matrices import DEFAULT_MATRIX_DTYPE, DEFAULT_REAL_DTYPE, IMAT, XMAT, ZMAT
from pyqtorch.utils import (
    ATOL,
    RTOL,
    is_normalized,
    operator_kron,
    overlap,
    product_state,
    random_state,
)

pi = torch.tensor(torch.pi)


def Hamiltonian(batch_size: int = 1) -> torch.Tensor:
    sigmaz = torch.diag(torch.tensor([1.0, -1.0], dtype=DEFAULT_MATRIX_DTYPE))
    Hbase = torch.kron(sigmaz, sigmaz)
    H = torch.kron(Hbase, Hbase)
    if batch_size == 1:
        return H
    elif batch_size == 2:
        return torch.stack((H, H.conj()), dim=2)
    else:
        raise NotImplementedError


def Hamiltonian_general(n_qubits: int = 2, batch_size: int = 1) -> torch.Tensor:
    H_batch = torch.zeros(
        (2**n_qubits, 2**n_qubits, batch_size), dtype=DEFAULT_MATRIX_DTYPE
    )
    for i in range(batch_size):
        H_0 = torch.randn((2**n_qubits, 2**n_qubits), dtype=DEFAULT_MATRIX_DTYPE)
        H = (H_0 + torch.conj(H_0.transpose(0, 1))).to(DEFAULT_MATRIX_DTYPE)
        H_batch[..., i] = H
    return H_batch


def Hamiltonian_diag(n_qubits: int = 2, batch_size: int = 1) -> torch.Tensor:
    H_batch = torch.zeros(
        (2**n_qubits, 2**n_qubits, batch_size), dtype=DEFAULT_MATRIX_DTYPE
    )
    for i in range(batch_size):
        H_0 = torch.randn((2**n_qubits, 2**n_qubits), dtype=DEFAULT_MATRIX_DTYPE)
        H = H_0 + torch.conj(H_0.transpose(0, 1))
        get_diag = torch.diag(H)
        H_diag = torch.diag(get_diag)
        H_batch[..., i] = H_diag
    return H_batch


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("n_qubits, batch_size", [(2, 1), (4, 2)])
def test_hamevo_general(n_qubits: int, batch_size: int) -> None:
    H = Hamiltonian_general(n_qubits, batch_size)
    t_evo = torch.rand(1, dtype=DEFAULT_REAL_DTYPE)
    hamevo = pyq.HamiltonianEvolution(H, t_evo, tuple([i for i in range(n_qubits)]))
    psi = random_state(n_qubits, batch_size)
    psi_star = hamevo(psi)
    assert is_normalized(psi_star, atol=ATOL)


@pytest.mark.flaky(max_runs=5)
def test_hamevo_single() -> None:
    n_qubits = 4
    H = Hamiltonian(1)
    t_evo = torch.tensor([torch.pi / 4], dtype=DEFAULT_MATRIX_DTYPE)
    hamevo = pyq.HamiltonianEvolution(H, t_evo, tuple([i for i in range(n_qubits)]))
    psi = pyq.uniform_state(n_qubits)
    psi_star = hamevo(psi)
    result = overlap(psi_star, psi)
    assert torch.isclose(
        result, torch.tensor([0.5], dtype=torch.float64), rtol=RTOL, atol=ATOL
    )


@pytest.mark.flaky(max_runs=5)
def test_hamevo_batch() -> None:
    n_qubits = 4
    batch_size = 2
    H = Hamiltonian(batch_size)
    t_evo = torch.tensor([torch.pi / 4], dtype=DEFAULT_MATRIX_DTYPE)
    hamevo = pyq.HamiltonianEvolution(H, t_evo, tuple([i for i in range(n_qubits)]))
    psi = pyq.uniform_state(n_qubits, batch_size)
    psi_star = hamevo(psi)
    result = overlap(psi_star, psi)
    assert torch.allclose(
        result, torch.tensor([0.5, 0.5], dtype=torch.float64), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize(
    "H, t_evo, target, batch_size",
    [
        (  # batchsize 1 | 1
            Hamiltonian(1),
            torch.tensor([torch.pi / 4], dtype=DEFAULT_MATRIX_DTYPE),
            torch.tensor([0.5], dtype=torch.float64),
            1,
        ),
        (  # batchsize 1 | 4
            Hamiltonian(1),
            torch.tensor([torch.pi / 4, 0.0, torch.pi / 2, torch.pi]),
            torch.tensor([0.5, 1.0, 0.0, 1.0], dtype=torch.float64),
            4,
        ),
        (  # batchsize 2 | 1
            Hamiltonian(2),
            torch.tensor([torch.pi / 4], dtype=DEFAULT_MATRIX_DTYPE),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            2,
        ),
        (  # batchsize 2 | 2
            Hamiltonian(2),
            torch.tensor([torch.pi / 4, torch.pi]),
            torch.tensor([0.5, 1.0], dtype=torch.float64),
            2,
        ),
    ],
)
def test_hamiltonianevolution_with_types(
    H: torch.Tensor,
    t_evo: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
) -> None:
    n_qubits = 4
    hamevo = pyq.HamiltonianEvolution(H, t_evo, tuple([i for i in range(n_qubits)]))
    psi = pyq.uniform_state(n_qubits)
    psi_star = hamevo(psi)
    result = overlap(psi_star, psi)
    assert result.size() == (batch_size,)
    assert torch.allclose(result, target, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n_qubits", [2])
def test_hevo_parametric_gen(n_qubits: int) -> None:
    dim = torch.randint(1, n_qubits + 1, (1,)).item()
    vparam = "theta"
    sup = tuple(range(dim))
    parametric = True
    ops = [pyq.X, pyq.Y, pyq.Z]
    # generator = pyq.Add([pyq.Scale(pyq.Z(0), vparam), pyq.Scale(pyq.Z(1), vparam)])
    qubit_targets = np.random.choice(dim, len(ops), replace=True)
    generator = pyq.Add([pyq.Scale(op(q), vparam) for op, q in zip(ops, qubit_targets)])
    hamevo = pyq.HamiltonianEvolution(generator, vparam, sup, parametric)
    vals = {"theta": torch.rand(1)}
    psi = random_state(n_qubits)
    psi_star = hamevo(psi, vals)
    psi_expected = _calc_mat_vec_wavefunction(hamevo, n_qubits, psi, vals)
    assert torch.allclose(psi_star, psi_expected, rtol=RTOL, atol=ATOL)


def test_hevo_constant_gen() -> None:
    sup = (0, 1)
    generator = pyq.Add(
        [pyq.Scale(pyq.Z(0), torch.rand(1)), pyq.Scale(pyq.Z(1), torch.rand(1))]
    )
    hamevo = pyq.HamiltonianEvolution(generator, torch.rand(1), sup)
    psi = pyq.zero_state(2)
    psi_star = hamevo(psi)
    psi_expected = _calc_mat_vec_wavefunction(hamevo, len(sup), psi)
    assert torch.allclose(psi_star, psi_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "H, t_evo, expected_state",
    [
        (
            Hamiltonian(1),
            torch.tensor([torch.pi / 4], dtype=DEFAULT_MATRIX_DTYPE),
            torch.tensor([0.5], dtype=torch.float64),
        ),
        (
            Hamiltonian(1),
            torch.tensor([torch.pi / 4], dtype=DEFAULT_MATRIX_DTYPE),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
        ),
        (
            Hamiltonian(1),
            torch.tensor([torch.pi / 4, torch.pi]),
            torch.tensor([0.5, 1.0], dtype=torch.float64),
        ),
    ],
)
def test_symbol_hamevo(
    H: torch.Tensor,
    t_evo: torch.Tensor,
    expected_state: torch.Tensor,
) -> None:
    symbol = "h"
    n_qubits = 4
    hamevo = pyq.HamiltonianEvolution(
        symbol, t_evo, tuple([i for i in range(n_qubits)])
    )
    psi = pyq.uniform_state(n_qubits)
    psi_star = hamevo(psi, {symbol: H})
    state = overlap(psi_star, psi)
    assert torch.allclose(state, expected_state, rtol=RTOL, atol=ATOL)


def test_hamevo_tensor() -> None:
    hermitian_matrix = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.complex128)
    hamiltonian_evolution = pyq.HamiltonianEvolution(
        generator=hermitian_matrix, time=torch.tensor([1.0j]), qubit_support=(0,)
    )

    expected_evo_result = torch.tensor(
        [[[13.1815 + 0.0j], [14.8839 + 0.0j]], [[14.8839 + 0.0j], [28.0655 + 0.0j]]],
        dtype=torch.complex128,
    )
    assert torch.allclose(
        hamiltonian_evolution.tensor(), expected_evo_result, atol=1.0e-4
    )


@pytest.mark.parametrize(
    "state_fn", [pyq.random_state, pyq.zero_state, pyq.uniform_state]
)
def test_parametric_phase_hamevo(
    state_fn: Callable, batch_size: int = 1, n_qubits: int = 1
) -> None:
    target = 0
    state = state_fn(n_qubits, batch_size=batch_size)
    phi = torch.rand(1, dtype=DEFAULT_MATRIX_DTYPE)
    H = (ZMAT - IMAT) / 2
    hamevo = pyq.HamiltonianEvolution(H, phi, (target,))
    phase = pyq.PHASE(target, "phi")
    assert torch.allclose(phase(state, {"phi": phi}), hamevo(state))


def test_hamevo_endianness() -> None:
    t = torch.ones(1)
    h = torch.tensor(
        [
            [0.9701, 0.0000, 0.7078, 0.0000],
            [0.0000, 0.9701, 0.0000, 0.7078],
            [0.4594, 0.0000, 0.9207, 0.0000],
            [0.0000, 0.4594, 0.0000, 0.9207],
        ],
        dtype=torch.complex128,
    )
    iszero = torch.tensor([False, True, False, True])
    op = pyq.HamiltonianEvolution(qubit_support=(0, 1), generator=h, time=t)
    st = op(pyq.zero_state(2)).flatten()
    assert torch.allclose(
        st[iszero], torch.zeros(1, dtype=DEFAULT_MATRIX_DTYPE), rtol=RTOL, atol=ATOL
    )

    h = torch.tensor(
        [
            [0.9701, 0.7078, 0.0000, 0.0000],
            [0.4594, 0.9207, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.9701, 0.7078],
            [0.0000, 0.0000, 0.4594, 0.9207],
        ],
        dtype=torch.complex128,
    )
    iszero = torch.tensor([False, False, True, True])
    op = pyq.HamiltonianEvolution(qubit_support=(0, 1), generator=h, time=t)
    st = op(pyq.zero_state(2)).flatten()
    assert torch.allclose(
        st[iszero], torch.zeros(1, dtype=DEFAULT_MATRIX_DTYPE), rtol=RTOL, atol=ATOL
    )


def test_hamevo_endianness_cnot() -> None:
    n_qubits = 2
    state_10 = product_state("10")

    gen = -0.5 * operator_kron((IMAT - ZMAT).unsqueeze(-1), (IMAT - XMAT).unsqueeze(-1))
    hamiltonian_evolution = pyq.HamiltonianEvolution(
        generator=gen,
        time=torch.tensor([torch.pi / 2.0]),
        qubit_support=tuple(range(n_qubits)),
    )
    wf_hamevo = hamiltonian_evolution(state_10)

    cnot_op = pyq.CNOT(0, 1)
    wf_cnot = cnot_op(state_10)
    assert torch.allclose(wf_cnot, wf_hamevo, rtol=RTOL, atol=ATOL)
