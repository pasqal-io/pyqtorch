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
@pytest.mark.parametrize("make_param", [True, False])
@pytest.mark.parametrize("commuting_terms", [False, True])
def test_hamevo_parametric_gen(
    n_qubits: int, batch_size: int, make_param: bool, commuting_terms: bool
) -> None:
    k_1q = 2 * n_qubits  # Number of 1-qubit terms
    k_2q = n_qubits**2  # Number of 2-qubit terms
    generator, param_list = random_pauli_hamiltonian(
        n_qubits,
        k_1q,
        k_2q,
        make_param=make_param,
        p_param=1.0,
        commuting_terms=commuting_terms,
    )
    tparam = "t"

    param_list.append("t")

    hamevo = pyq.HamiltonianEvolution(generator, tparam, cache_length=2)

    if make_param:
        assert hamevo.generator_type in (
            GeneratorType.PARAMETRIC_OPERATION,
            GeneratorType.PARAMETRIC_COMMUTING_SEQUENCE,
        )
    else:
        assert hamevo.generator_type in (
            GeneratorType.OPERATION,
            GeneratorType.COMMUTING_SEQUENCE,
        )
    assert len(hamevo._cache_hamiltonian_evo) == 0

    def apply_hamevo_and_compare_expected(psi, values):
        psi_star = hamevo(psi, values)
        psi_expected = calc_mat_vec_wavefunction(hamevo, psi, values)
        assert torch.allclose(psi_star, psi_expected, rtol=RTOL, atol=ATOL)

    values = {param: torch.rand(batch_size) for param in param_list}

    psi = random_state(n_qubits)
    apply_hamevo_and_compare_expected(psi, values)

    # test cached
    if not commuting_terms:
        assert len(hamevo._cache_hamiltonian_evo) == 1
    apply_hamevo_and_compare_expected(psi, values)
    if not commuting_terms:
        assert len(hamevo._cache_hamiltonian_evo) == 1

    # test caching new value
    for param in param_list:
        values[param] += 0.1

    apply_hamevo_and_compare_expected(psi, values)
    if not commuting_terms:
        assert len(hamevo._cache_hamiltonian_evo) == 2

    # changing input state should not change the cache
    psi = random_state(n_qubits)
    apply_hamevo_and_compare_expected(psi, values)
    if not commuting_terms:
        assert len(hamevo._cache_hamiltonian_evo) == 2

    # test limit cache
    previous_cache_keys = hamevo._cache_hamiltonian_evo.keys()

    for param in param_list:
        values[param] += 0.1

    values_cache_key = str(OrderedDict(values))
    if not commuting_terms:
        assert values_cache_key not in previous_cache_keys

    apply_hamevo_and_compare_expected(psi, values)
    if not commuting_terms:
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


@pytest.mark.parametrize("batch_size", [1, 2])
def test_hamiltonian_evolution_parameter_reembedding_complex(batch_size: int) -> None:
    """
    Test parameter re-embedding with complex nested expressions.

    This test creates a more complex parameter dependency chain similar to
    the expressions mentioned in pasqal-io/qadence#492:

    Mathematical structure:
    base_expr = x * theta
    trig_expr = sin(base_expr * t)
    final_coeff = (omega * y * trig_expr) + (alpha * cos(t))
    H(t) = final_coeff * σₓ

    Verifies that re-embedding correctly handles:
    1. Nested ConcretizedCallable dependencies
    2. Multiple levels of parameter composition
    3. Mixed time-dependent and time-independent parameters
    """
    tparam = "t"

    # Create nested parameter expressions using BINARY operations only
    # Layer 1: base computation
    base_expr, base_fn = "base_expr", ConcretizedCallable("mul", ["x", "theta"])

    # Layer 2: time-dependent expressions
    base_times_t, base_times_t_fn = (
        "base_times_t",
        ConcretizedCallable("mul", ["base_expr", tparam]),
    )
    sin_base_t, sin_base_t_fn = "sin_base_t", ConcretizedCallable(
        "sin", ["base_times_t"]
    )
    cos_t, cos_t_fn = "cos_t", ConcretizedCallable("cos", [tparam])

    # Layer 3: binary multiplication chains
    omega_y, omega_y_fn = "omega_y", ConcretizedCallable("mul", ["omega", "y"])
    term1, term1_fn = "term1", ConcretizedCallable("mul", ["omega_y", "sin_base_t"])
    term2, term2_fn = "term2", ConcretizedCallable("mul", ["alpha", "cos_t"])
    final_coeff, final_coeff_fn = "final_coeff", ConcretizedCallable(
        "add", ["term1", "term2"]
    )

    # Create generator: H(t) = final_coeff * X(0)
    generator = Scale(pyq.X(0), final_coeff)

    # Create embedding with all parameter mappings
    embedding = pyq.Embedding(
        vparam_names=["x", "theta", "y"],  # Variational parameters
        fparam_names=["omega", "alpha"],  # Feature parameters
        var_to_call={
            base_expr: base_fn,
            base_times_t: base_times_t_fn,
            sin_base_t: sin_base_t_fn,
            cos_t: cos_t_fn,
            omega_y: omega_y_fn,
            term1: term1_fn,
            term2: term2_fn,
            final_coeff: final_coeff_fn,
        },
        tparam_name=tparam,
    )

    # Create HamiltonianEvolution
    hamevo = pyq.HamiltonianEvolution(
        generator=generator, time=tparam, qubit_support=(0,)
    )

    # Test parameters - INCLUDE time parameter from the start
    values = {
        "x": torch.tensor(1.5),
        "theta": torch.tensor(0.8),
        "y": torch.tensor(2.0),
        "omega": torch.tensor(1.2),
        "alpha": torch.tensor(0.7),
        tparam: torch.tensor(0.0),
    }

    # Test different time values
    test_times = [0.1, 0.5, 1.0, 2.0]

    for t_val in test_times:
        # Full embedding approach
        values_full = values.copy()
        values_full[tparam] = torch.tensor(t_val)
        embedded_full = embedding.embed_all(values_full.copy())

        # Re-embedding approach
        embedded_base = embedding.embed_all(values.copy())
        embedded_reembed = embedding.reembed_tparam(
            embedded_base.copy(), torch.tensor(t_val)
        )

        # Get evolution operators
        op_full = hamevo.tensor(values=embedded_full, embedding=None)
        op_reembed = hamevo.tensor(values=embedded_reembed, embedding=None)

        # Verify equivalence
        assert torch.allclose(
            op_full, op_reembed, rtol=1e-12, atol=1e-14
        ), f"Complex re-embedding failed for t={t_val}"

        # Verify that all time-dependent parameters are tracked
        expected_tracked = {base_times_t, sin_base_t, cos_t, term1, term2, final_coeff}
        actual_tracked = set(embedding.tracked_vars)
        assert expected_tracked <= actual_tracked, (
            f"Missing tracked vars: expected {expected_tracked}, "
            f"got {actual_tracked}"
        )


def test_hamiltonian_evolution_parameter_reembedding_state_evolution() -> None:
    """
    Test that parameter re-embedding produces identical quantum state evolution.

    This test verifies the complete quantum evolution process, not just the
    tensor construction, ensuring that re-embedding doesn't introduce any
    phase errors or unitary violations in the time evolution.
    """
    tparam = "t"
    n_qubits = 2

    # Create time-dependent generator: H(t) = (ω·t)·cos(ω·t)·(X⊗I + I⊗Z)
    # Using binary operations: (omega * t) * cos(omega * t)
    omega_t, omega_t_fn = "omega_t", ConcretizedCallable("mul", ["omega", tparam])
    cos_omega_t, cos_omega_t_fn = "cos_omega_t", ConcretizedCallable("cos", ["omega_t"])
    omega_times_t, omega_times_t_fn = "omega_times_t", ConcretizedCallable(
        "mul", ["omega", tparam]
    )
    coeff, coeff_fn = "coeff", ConcretizedCallable(
        "mul", ["omega_times_t", "cos_omega_t"]
    )

    generator = Scale(Add([pyq.X(0), pyq.Z(1)]), coeff)  # X⊗I + I⊗Z

    embedding = pyq.Embedding(
        fparam_names=["omega"],
        var_to_call={
            omega_t: omega_t_fn,
            cos_omega_t: cos_omega_t_fn,
            omega_times_t: omega_times_t_fn,
            coeff: coeff_fn,
        },
        tparam_name=tparam,
    )

    hamevo = pyq.HamiltonianEvolution(
        generator=generator, time=tparam, qubit_support=tuple(range(n_qubits))
    )

    # Initial state |1⟩⊗|0⟩
    initial_state = pyq.product_state("10")  # |1⟩⊗|0⟩
    omega_val = torch.tensor(1.5)

    # Test multiple evolution times
    evolution_times = [0.1, 0.5, 1.0]

    for t_val in evolution_times:
        # Method 1: Full embedding - INCLUDE time parameter from start
        values_full = {"omega": omega_val, tparam: torch.tensor(t_val)}
        embedded_full = embedding.embed_all(values_full.copy())
        final_state_full = hamevo(initial_state, values=embedded_full, embedding=None)

        # Method 2: Re-embedding - INCLUDE time parameter from start
        values_base = {"omega": omega_val, tparam: torch.tensor(0.0)}
        embedded_base = embedding.embed_all(values_base.copy())
        embedded_reembed = embedding.reembed_tparam(
            embedded_base.copy(), torch.tensor(t_val)
        )
        final_state_reembed = hamevo(
            initial_state, values=embedded_reembed, embedding=None
        )

        # Verify state equivalence
        assert torch.allclose(
            final_state_full, final_state_reembed, rtol=1e-12, atol=1e-14
        ), f"State evolution differs for t={t_val}"

        # Verify normalization is preserved
        norm_full = torch.sum(torch.abs(final_state_full.flatten()) ** 2)
        norm_reembed = torch.sum(torch.abs(final_state_reembed.flatten()) ** 2)
        # Fix dtype mismatch: use same dtype as the computed norm
        expected_norm = torch.tensor(1.0, dtype=norm_full.dtype)
        assert torch.allclose(norm_full, expected_norm, atol=1e-12)
        assert torch.allclose(norm_reembed, expected_norm, atol=1e-12)


def test_hamiltonian_evolution_parameter_reembedding_edge_cases() -> None:
    """
    Test edge cases for parameter re-embedding to ensure robustness.

    Edge cases tested:
    1. Parameters that don't depend on time (should not be in tracked_vars)
    2. Zero time evolution
    3. Parameters with constant values
    4. Empty parameter dependencies
    """
    tparam = "t"

    # Case 1: Mixed time-dependent and time-independent parameters
    sin_t, sin_t_fn = "sin_t", ConcretizedCallable("sin", [tparam])
    const_expr, const_expr_fn = "const_expr", ConcretizedCallable("mul", ["a", "b"])

    generator = Add(
        [
            Scale(pyq.X(0), sin_t),  # Time-dependent
            Scale(pyq.Z(0), const_expr),  # Time-independent
        ]
    )

    embedding = pyq.Embedding(
        fparam_names=["a", "b"],
        var_to_call={
            sin_t: sin_t_fn,
            const_expr: const_expr_fn,
        },
        tparam_name=tparam,
    )

    hamevo = pyq.HamiltonianEvolution(generator, tparam, qubit_support=(0,))

    values = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}

    # Initialize embedding to trigger tracking analysis
    values_init = values.copy()
    values_init[tparam] = torch.tensor(0.0)
    embedded_base = embedding.embed_all(values_init)

    # Verify tracking: only sin_t should be tracked
    assert sin_t in embedding.tracked_vars, "sin_t should be tracked (depends on time)"
    assert (
        const_expr not in embedding.tracked_vars
    ), "const_expr should not be tracked (time-independent)"
    # Case 2: Zero time evolution should work
    embedded_zero = embedding.reembed_tparam(embedded_base.copy(), torch.tensor(0.0))
    op_zero = hamevo.tensor(values=embedded_zero, embedding=None)

    # Should be close to identity for small times
    assert torch.allclose(
        torch.diagonal(op_zero, dim1=0, dim2=1).squeeze(),
        torch.tensor([1.0 + 0j, 1.0 + 0j], dtype=torch.complex128),
        rtol=1e-10,
        atol=1e-12,
    )

    # Case 3: Large time values should not cause numerical instability
    large_t = torch.tensor(100.0)
    embedded_large = embedding.reembed_tparam(embedded_base.copy(), large_t)
    op_large = hamevo.tensor(values=embedded_large, embedding=None)

    # Should still be unitary
    op_dagger = torch.conj(op_large.transpose(0, 1))
    identity_check = torch.matmul(op_large.squeeze(), op_dagger.squeeze())
    expected_identity = torch.eye(2, dtype=torch.complex128)
    assert torch.allclose(identity_check, expected_identity, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("n_time_steps", [5, 10])
def test_hamiltonian_evolution_parameter_reembedding_performance_verification(
    n_time_steps: int,
) -> None:
    """
    Test that re-embedding maintains mathematical equivalence across many time steps.

    This test simulates a scenario where the same Hamiltonian is evaluated at many
    different time points, which is typical in time-dependent evolution. It verifies
    that re-embedding produces identical results to full embedding at every step.
    """
    tparam = "t"

    # Create a moderately complex time-dependent expression
    # H(t) = [ω₁·sin(ωt) + ω₂·cos(ωt)]·X + [β·t²]·Z
    omega_t, omega_t_fn = "omega_t", ConcretizedCallable("mul", ["omega", tparam])
    sin_omega_t, sin_omega_t_fn = "sin_omega_t", ConcretizedCallable("sin", ["omega_t"])
    cos_omega_t, cos_omega_t_fn = "cos_omega_t", ConcretizedCallable("cos", ["omega_t"])
    t_squared, t_squared_fn = "t_squared", ConcretizedCallable("mul", [tparam, tparam])

    term1, term1_fn = "term1", ConcretizedCallable("mul", ["omega1", "sin_omega_t"])
    term2, term2_fn = "term2", ConcretizedCallable("mul", ["omega2", "cos_omega_t"])
    x_coeff, x_coeff_fn = "x_coeff", ConcretizedCallable("add", ["term1", "term2"])
    z_coeff, z_coeff_fn = "z_coeff", ConcretizedCallable("mul", ["beta", "t_squared"])

    generator = Add(
        [
            Scale(pyq.X(0), x_coeff),
            Scale(pyq.Z(0), z_coeff),
        ]
    )

    embedding = pyq.Embedding(
        fparam_names=["omega", "omega1", "omega2", "beta"],
        var_to_call={
            omega_t: omega_t_fn,
            sin_omega_t: sin_omega_t_fn,
            cos_omega_t: cos_omega_t_fn,
            t_squared: t_squared_fn,
            term1: term1_fn,
            term2: term2_fn,
            x_coeff: x_coeff_fn,
            z_coeff: z_coeff_fn,
        },
        tparam_name=tparam,
    )

    hamevo = pyq.HamiltonianEvolution(generator, tparam, qubit_support=(0,))

    # Parameters
    params = {
        "omega": torch.tensor(2.0),
        "omega1": torch.tensor(1.5),
        "omega2": torch.tensor(0.8),
        "beta": torch.tensor(0.3),
    }

    # Generate time points
    time_points = torch.linspace(0.0, 2.0, n_time_steps)

    # Initialize base embedding
    params_base = params.copy()
    params_base[tparam] = torch.tensor(0.0)
    embedded_base = embedding.embed_all(params_base.copy())

    max_difference = 0.0

    for t_val in time_points:
        # Full embedding
        params_full = params.copy()
        params_full[tparam] = t_val
        embedded_full = embedding.embed_all(params_full.copy())
        op_full = hamevo.tensor(values=embedded_full, embedding=None)

        # Re-embedding
        embedded_reembed = embedding.reembed_tparam(embedded_base.copy(), t_val)
        op_reembed = hamevo.tensor(values=embedded_reembed, embedding=None)

        # Track maximum difference
        diff = torch.max(torch.abs(op_full - op_reembed))
        max_difference = max(max_difference, diff.item())

        # Verify equivalence at each step
        assert torch.allclose(
            op_full, op_reembed, rtol=1e-12, atol=1e-14
        ), f"Difference found at t={t_val}: {diff}"

    # Ensure overall precision is maintained
    assert (
        max_difference < 1e-13
    ), f"Maximum difference {max_difference} exceeds tolerance"

    # Verify that all expected parameters are tracked
    expected_tracked = {
        omega_t,
        sin_omega_t,
        cos_omega_t,
        t_squared,
        term1,
        term2,
        x_coeff,
        z_coeff,
    }
    actual_tracked = set(embedding.tracked_vars)
    assert (
        expected_tracked <= actual_tracked
    ), f"Missing tracked parameters: {expected_tracked - actual_tracked}"


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("n_qubits", [2, 3])
def test_hamiltonian_evolution_parameter_reembedding_basic(
    batch_size: int, n_qubits: int
) -> None:
    """
    Test basic parameter re-embedding functionality in time-dependent HamiltonianEvolution.

    This test verifies that reembed_tparam produces identical results to full embedding
    for a simple time-dependent generator using sin(t) and t^2 terms.

    Mathematical verification:
    H(t) = ω[y·sin(t)·σₓ⊗I + x·t²·I⊗σᵧ]

    The test ensures that:
    embedding.embed_all() ≡ embedding.reembed_tparam()
    for different time values, up to numerical precision.
    """
    # Create time-dependent parameter expressions using ConcretizedCallable
    tparam = "t"
    sin_t, sin_t_fn = "sin_t", ConcretizedCallable("sin", [tparam])
    t_sq, t_sq_fn = "t_sq", ConcretizedCallable("mul", [tparam, tparam])

    # Create a complex generator with nested dependencies
    # H(t) = ω[y·sin(t)·X(0) + x·t²·Y(1)]
    omega = 2.5
    generator = Scale(
        Add(
            [
                Scale(Scale(pyq.X(0), sin_t), "y"),  # y·sin(t)·X(0)
                Scale(Scale(pyq.Y(1), t_sq), "x"),  # x·t²·Y(1)
            ]
        ),
        omega,
    )

    # Create embedding with time parameter tracking
    embedding = pyq.Embedding(
        vparam_names=["x", "y"],  # Variational parameters
        fparam_names=[],  # No feature parameters
        var_to_call={sin_t: sin_t_fn, t_sq: t_sq_fn},  # Time-dependent mappings
        tparam_name=tparam,  # Track time parameter
    )

    # Create HamiltonianEvolution
    hamevo = pyq.HamiltonianEvolution(
        generator=generator, time=tparam, qubit_support=tuple(range(n_qubits))
    )

    # Test parameters - INCLUDE time parameter from the start
    x_val = torch.rand(batch_size, requires_grad=True)
    y_val = torch.rand(batch_size, requires_grad=True)
    base_values = {"x": x_val, "y": y_val, tparam: torch.tensor(0.0)}  # Include time

    # Test multiple time values to ensure robustness
    test_times = [0.0, 0.5, 1.0, 1.5708, 3.14159]  # Include 0, π/2, π

    for t_val in test_times:
        values_full = base_values.copy()
        values_full[tparam] = torch.tensor(t_val)

        # Method 1: Full embedding (recalculates everything)
        embedded_full = embedding.embed_all(values_full.copy())
        evolved_op_full = hamevo.tensor(values=embedded_full, embedding=None)

        # Method 2: Re-embedding (only recalculates time-dependent parameters)
        embedded_base = embedding.embed_all(base_values.copy())
        embedded_reembed = embedding.reembed_tparam(
            embedded_base.copy(), torch.tensor(t_val)
        )
        evolved_op_reembed = hamevo.tensor(values=embedded_reembed, embedding=None)

        # Verify mathematical equivalence
        assert torch.allclose(
            evolved_op_full, evolved_op_reembed, rtol=1e-12, atol=1e-14
        ), (
            f"Re-embedding failed for t={t_val}. "
            f"Max diff: {torch.max(torch.abs(evolved_op_full - evolved_op_reembed))}"
        )
        # Verify that tracked variables are correctly identified
        expected_tracked = {sin_t, t_sq}  # Both depend on time
        assert (
            set(embedding.tracked_vars) == expected_tracked
        ), f"Expected tracked vars {expected_tracked}, got {set(embedding.tracked_vars)}"
