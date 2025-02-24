from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import pytest
import torch
from helpers import calc_mat_vec_wavefunction, random_pauli_hamiltonian

import pyqtorch as pyq
from pyqtorch import RX, Add, ConcretizedCallable, HamiltonianEvolution, Scale, X
from pyqtorch.composite import Sequence
from pyqtorch.hamiltonians import GeneratorType
from pyqtorch.matrices import (
    DEFAULT_MATRIX_DTYPE,
    DEFAULT_REAL_DTYPE,
    IMAT,
    XMAT,
    ZMAT,
)
from pyqtorch.noise import AnalogDepolarizing
from pyqtorch.utils import (
    ATOL,
    RTOL,
    DensityMatrix,
    SolverType,
    density_mat,
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
    assert len(hamevo._cache_hamiltonian_evo) == 0
    psi = pyq.uniform_state(n_qubits)
    psi_star = hamevo(psi)
    result = overlap(psi_star, psi)
    assert result.size() == (batch_size,)
    assert torch.allclose(result, target, rtol=RTOL, atol=ATOL)

    # test cached
    assert len(hamevo._cache_hamiltonian_evo) == 1

    psi_star = hamevo(psi)
    assert len(hamevo._cache_hamiltonian_evo) == 1
    result = overlap(psi_star, psi)
    # check value has not changed
    assert torch.allclose(result, target, rtol=RTOL, atol=ATOL)


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
        # with batchdim
        (
            Hamiltonian(2),
            torch.tensor([torch.pi / 4], dtype=DEFAULT_MATRIX_DTYPE),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
        ),
        # with batchdim in position 0
        (
            Hamiltonian(2).transpose(0, 2),
            torch.tensor([torch.pi / 4], dtype=DEFAULT_MATRIX_DTYPE),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
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
    assert hamevo.generator_type == GeneratorType.SYMBOL
    psi = pyq.uniform_state(n_qubits)
    psi_star = hamevo(psi, {symbol: H})
    state = overlap(psi_star, psi)
    assert torch.allclose(state, expected_state, rtol=RTOL, atol=ATOL)


def test_hamevo_fixed_tensor_result() -> None:
    hermitian_matrix = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.complex128)
    hamevo = pyq.HamiltonianEvolution(
        generator=hermitian_matrix, time=torch.tensor([1.0j]), qubit_support=(0,)
    )
    assert hamevo.generator_type == GeneratorType.TENSOR
    expected_evo_result = torch.tensor(
        [[[13.1815 + 0.0j], [14.8839 + 0.0j]], [[14.8839 + 0.0j], [28.0655 + 0.0j]]],
        dtype=torch.complex128,
    )
    assert torch.allclose(hamevo.tensor(), expected_evo_result, atol=1.0e-4)


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
    assert hamevo.generator_type == GeneratorType.TENSOR
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


@pytest.mark.parametrize("duration", [torch.rand(1), "duration"])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("use_sparse", [True, False])
@pytest.mark.parametrize("ode_solver", [SolverType.DP5_SE, SolverType.KRYLOV_SE])
def test_timedependent(
    tparam: str,
    param_y: float,
    duration: float,
    batch_size: int,
    use_sparse: bool,
    n_steps: int,
    torch_hamiltonian: Callable,
    hamevo_generator: Sequence,
    sin: tuple,
    sq: tuple,
    ode_solver: SolverType,
) -> None:

    psi_start = random_state(2, batch_size)

    dur_val = duration if isinstance(duration, torch.Tensor) else torch.rand(1)

    # simulate with time-dependent solver
    t_points = torch.linspace(0, dur_val[0], n_steps)

    psi_solver = pyq.sesolve(
        torch_hamiltonian,
        psi_start.reshape(-1, batch_size),
        t_points,
        ode_solver,
        options={"use_sparse": use_sparse},
    ).states[-1]

    # simulate with HamiltonianEvolution
    embedding = pyq.Embedding(
        tparam_name=tparam,
        var_to_call={sin[0]: sin[1], sq[0]: sq[1]},
    )
    hamiltonian_evolution = pyq.HamiltonianEvolution(
        generator=hamevo_generator,
        time=tparam,
        duration=dur_val,
        steps=n_steps,
        solver=ode_solver,
        use_sparse=use_sparse,
    )
    values = {"y": param_y, "duration": dur_val}
    psi_hamevo = hamiltonian_evolution(
        state=psi_start, values=values, embedding=embedding
    ).reshape(-1, batch_size)

    assert torch.allclose(psi_solver, psi_hamevo, rtol=RTOL, atol=1.0e-3)


@pytest.mark.parametrize("duration", [torch.rand(1), "duration"])
@pytest.mark.parametrize(
    "batchsize",
    [
        1,
        3,
    ],
)
def test_timedependent_with_noise(
    tparam: str,
    param_y: float,
    duration: float,
    batchsize: int,
    n_steps: int,
    torch_hamiltonian: Callable,
    hamevo_generator: Sequence,
    sin: tuple,
    sq: tuple,
) -> None:

    psi_start = density_mat(random_state(2, batchsize))
    dur_val = duration if isinstance(duration, torch.Tensor) else torch.rand(1)

    # simulate with time-dependent solver
    t_points = torch.linspace(0, dur_val[0], n_steps)

    # Define jump operators
    # Note that we squeeze to remove the batch dimension
    noise = AnalogDepolarizing(error_param=0.1, qubit_support=0)
    list_ops = noise._noise_operators(full_support=(0, 1))
    solver = SolverType.DP5_ME
    psi_solver = pyq.mesolve(
        torch_hamiltonian, psi_start, list_ops, t_points, solver
    ).states[-1]

    # simulate with HamiltonianEvolution
    embedding = pyq.Embedding(
        tparam_name=tparam,
        var_to_call={sin[0]: sin[1], sq[0]: sq[1]},
    )
    hamiltonian_evolution = pyq.HamiltonianEvolution(
        generator=hamevo_generator,
        time=tparam,
        duration=duration,
        steps=n_steps,
        solver=solver,
        noise=list_ops,
    )
    values = {"y": param_y, "duration": dur_val}
    psi_hamevo = hamiltonian_evolution(
        state=psi_start, values=values, embedding=embedding
    )
    assert isinstance(psi_hamevo, DensityMatrix)
    assert torch.allclose(psi_solver, psi_hamevo, rtol=RTOL, atol=1.0e-3)


def test_error_noise_qubit_support(
    tparam: str,
    duration: float,
    n_steps: int,
    hamevo_generator: Sequence,
):
    solver = SolverType.DP5_ME
    with pytest.raises(ValueError):
        noise = AnalogDepolarizing(error_param=0.1, qubit_support=3)
        hamevo = pyq.HamiltonianEvolution(
            generator=hamevo_generator,
            time=tparam,
            duration=duration,
            steps=n_steps,
            solver=solver,
            noise=noise,
        )


@pytest.mark.parametrize("n_qubits", [2, 4, 6])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_hamevo_parametric_gen(n_qubits: int, batch_size: int) -> None:
    k_1q = 2 * n_qubits  # Number of 1-qubit terms
    k_2q = n_qubits**2  # Number of 2-qubit terms
    generator, param_list = random_pauli_hamiltonian(
        n_qubits, k_1q, k_2q, make_param=True, p_param=1.0
    )

    tparam = "t"

    param_list.append("t")

    hamevo = pyq.HamiltonianEvolution(generator, tparam, cache_length=2)

    assert hamevo.generator_type == GeneratorType.PARAMETRIC_OPERATION
    assert len(hamevo._cache_hamiltonian_evo) == 0

    def apply_hamevo_and_compare_expected(psi, values):
        psi_star = hamevo(psi, values)
        psi_expected = calc_mat_vec_wavefunction(hamevo, psi, values)
        assert torch.allclose(psi_star, psi_expected, rtol=RTOL, atol=ATOL)

    values = {param: torch.rand(batch_size) for param in param_list}

    psi = random_state(n_qubits)
    apply_hamevo_and_compare_expected(psi, values)

    # test cached
    assert len(hamevo._cache_hamiltonian_evo) == 1
    apply_hamevo_and_compare_expected(psi, values)
    assert len(hamevo._cache_hamiltonian_evo) == 1

    # test caching new value
    for param in param_list:
        values[param] += 0.1

    apply_hamevo_and_compare_expected(psi, values)
    assert len(hamevo._cache_hamiltonian_evo) == 2

    # changing input state should not change the cache
    psi = random_state(n_qubits)
    apply_hamevo_and_compare_expected(psi, values)
    assert len(hamevo._cache_hamiltonian_evo) == 2

    # test limit cache
    previous_cache_keys = hamevo._cache_hamiltonian_evo.keys()

    for param in param_list:
        values[param] += 0.1

    values_cache_key = str(OrderedDict(values))
    assert values_cache_key not in previous_cache_keys

    apply_hamevo_and_compare_expected(psi, values)
    assert len(hamevo._cache_hamiltonian_evo) == 2
    assert values_cache_key in previous_cache_keys


@pytest.mark.parametrize(
    "generator, time_param, result",
    [
        (RX(0, "x"), "x", True),
        (RX(1, 0.5), "y", False),
        (RX(0, "x"), "y", False),
        (RX(0, "x"), torch.tensor(0.5), False),
        (RX(0, torch.tensor(0.5)), torch.tensor(0.5), False),
        (Scale(X(1), "y"), "y", True),
        (Scale(X(1), 0.2), "x", False),
        (
            Add(
                [Scale(X(1), ConcretizedCallable("mul", ["y", "x"])), Scale(X(1), "z")]
            ),
            "x",
            True,
        ),
        (
            Add(
                [Scale(X(1), ConcretizedCallable("add", ["y", "x"])), Scale(X(1), "z")]
            ),
            "t",
            False,
        ),
    ],
)
def test_hamevo_is_time_dependent_generator(generator, time_param, result) -> None:
    hamevo = HamiltonianEvolution(generator, time_param)
    assert hamevo.is_time_dependent == result
