from __future__ import annotations

import pytest
import torch
from helpers import random_pauli_hamiltonian

import pyqtorch as pyq
from pyqtorch import DiffMode, expectation
from pyqtorch.matrices import COMPLEX_TO_REAL_DTYPES
from pyqtorch.primitives import Primitive
from pyqtorch.utils import (
    ATOL,
    GRADCHECK_ATOL,
    GRADCHECK_ATOL_hamevo,
    density_mat,
)


def circuit_example(
    n_qubits: int, n_layers: int
) -> tuple[pyq.QuantumCircuit, dict[str, torch.Tensor]]:
    rx = pyq.RX(0, param_name="theta_0")
    cry = pyq.CPHASE(0, 1, param_name="theta_1")
    rz = pyq.RZ(2, param_name="theta_2")

    cnot = pyq.CNOT(1, 2)
    ops = [rx, cry, rz, cnot] * n_layers
    circ = pyq.QuantumCircuit(n_qubits, ops)

    theta_0_value = torch.pi / 2
    theta_1_value = torch.pi
    theta_2_value = torch.pi / 4
    values = {
        "theta_0": torch.tensor([theta_0_value], requires_grad=True),
        "theta_1": torch.tensor([theta_1_value], requires_grad=True),
        "theta_2": torch.tensor([theta_2_value], requires_grad=True),
    }
    return circ, values


@pytest.mark.parametrize("n_qubits", [3, 4, 5])
@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_ad_observable(
    n_qubits: int,
    n_layers: int,
) -> None:
    circ, values_circuit = circuit_example(n_qubits, n_layers)
    state = pyq.zero_state(n_qubits)
    obs = pyq.Observable(pyq.RZ(0, "theta_obs"))
    assert obs.is_parametric
    assert obs.params == ["theta_obs"]
    values_obs = {"theta_obs": torch.tensor([torch.pi / 4], requires_grad=True)}
    values_all = values_circuit | values_obs

    exp_ad = expectation(circ, state, values_all, obs, DiffMode.AD)
    grad_ad = torch.autograd.grad(
        exp_ad, tuple(values_all.values()), torch.ones_like(exp_ad)
    )
    assert len(grad_ad) == len(values_circuit) + 1

    values_separated = {"circuit": values_circuit, "observables": values_obs}
    exp_ad_separate = expectation(circ, state, values_separated, obs, DiffMode.AD)
    assert torch.allclose(exp_ad_separate, exp_ad)
    grad_ad_obs = torch.autograd.grad(
        exp_ad_separate, tuple(values_obs.values()), torch.ones_like(exp_ad)
    )
    assert len(grad_ad_obs) == len(obs.params)
    assert torch.allclose(grad_ad[-1], grad_ad_obs[0])


@pytest.mark.parametrize("n_qubits", [3, 4, 5])
@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_adjoint_diff(n_qubits: int, n_layers: int) -> None:
    circ, values_ad = circuit_example(n_qubits, n_layers)
    obs = pyq.Observable(pyq.Z(0))
    assert not obs.is_parametric

    state = pyq.zero_state(n_qubits)
    values_adjoint = values_ad.copy()

    exp_ad = expectation(circ, state, values_ad, obs, DiffMode.AD)
    exp_adjoint = expectation(circ, state, values_adjoint, obs, DiffMode.ADJOINT)

    grad_ad = torch.autograd.grad(
        exp_ad, tuple(values_ad.values()), torch.ones_like(exp_ad)
    )
    assert len(grad_ad) == len(values_ad)

    grad_adjoint = torch.autograd.grad(
        exp_adjoint, tuple(values_adjoint.values()), torch.ones_like(exp_adjoint)
    )

    assert len(grad_ad) == len(grad_adjoint)
    for i in range(len(grad_ad)):
        if grad_adjoint[i] is not None:
            assert torch.allclose(grad_ad[i], grad_adjoint[i], atol=GRADCHECK_ATOL)

    # TODO higher order adjoint is not yet supported.
    # gradgrad_adjoint = torch.autograd.grad(
    #     grad_adjoint, tuple(values_adjoint.values()), torch.ones_like(grad_adjoint)
    # )


@pytest.mark.parametrize("n_qubits", [3, 4, 5])
@pytest.mark.parametrize("n_layers", [1, 2, 3])
@pytest.mark.parametrize("diagonal", [True, False])
@pytest.mark.parametrize("commuting_terms", [False, True])
def test_adjoint_diff_hamevo(
    n_qubits: int, n_layers: int, diagonal: bool, commuting_terms: bool
) -> None:

    cnot = pyq.CNOT(1, 2)
    ham = random_pauli_hamiltonian(
        n_qubits,
        k_1q=n_qubits,
        k_2q=0,
        default_scale_coeffs=1.0,
        diagonal=diagonal,
        commuting_terms=commuting_terms,
    )[0]
    ham_op = pyq.HamiltonianEvolution(ham, "t", qubit_support=tuple(range(n_qubits)))
    ops = [ham_op, cnot] * n_layers
    circ = pyq.QuantumCircuit(n_qubits, ops)
    obs = pyq.Observable(pyq.Z(0))

    t = torch.pi / 3

    state = pyq.zero_state(n_qubits)

    t_ad = torch.tensor([t], requires_grad=True)
    t_adjoint = torch.tensor([t], requires_grad=True)

    values_ad = {"t": t_ad}
    values_adjoint = {
        "t": t_adjoint,
    }
    exp_ad = expectation(circ, state, values_ad, obs, DiffMode.AD)
    exp_adjoint = expectation(circ, state, values_adjoint, obs, DiffMode.ADJOINT)

    grad_ad = torch.autograd.grad(
        exp_ad, tuple(values_ad.values()), torch.ones_like(exp_ad)
    )

    grad_adjoint = torch.autograd.grad(
        exp_adjoint, tuple(values_adjoint.values()), torch.ones_like(exp_adjoint)
    )

    assert len(grad_ad) == len(grad_adjoint)
    for i in range(len(grad_ad)):
        assert torch.allclose(grad_ad[i], grad_adjoint[i], atol=GRADCHECK_ATOL_hamevo)


@pytest.mark.parametrize("n_qubits", [3, 5])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("ops_op", [pyq.Z, pyq.Y])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_sampled_diff(
    n_qubits: int,
    batch_size: int,
    ops_op: Primitive,
    dtype: torch.dtype,
) -> None:
    rx = pyq.RX(0, param_name="theta_0")
    cry = pyq.CPHASE(0, 1, param_name="theta_1")
    rz = pyq.RZ(2, param_name="theta_2")
    cnot = pyq.CNOT(1, 2)
    ops = [rx, cry, rz, cnot]
    circ = pyq.QuantumCircuit(n_qubits, ops).to(dtype)
    obs = pyq.Observable(pyq.Add([ops_op(i) for i in range(n_qubits)])).to(dtype)

    theta_0_value = torch.pi / 2
    theta_1_value = torch.pi
    theta_2_value = torch.pi / 4

    state = pyq.random_state(n_qubits, batch_size=batch_size, dtype=dtype)
    theta_0_ad = torch.tensor(
        [theta_0_value], requires_grad=True, dtype=COMPLEX_TO_REAL_DTYPES[dtype]
    )
    theta_1_ad = torch.tensor(
        [theta_1_value], requires_grad=True, dtype=COMPLEX_TO_REAL_DTYPES[dtype]
    )
    theta_2_ad = torch.tensor(
        [theta_2_value], requires_grad=True, dtype=COMPLEX_TO_REAL_DTYPES[dtype]
    )
    values = {"theta_0": theta_0_ad, "theta_1": theta_1_ad, "theta_2": theta_2_ad}

    exp_ad = expectation(circ, state, values, obs, DiffMode.AD)
    exp_ad_sampled = expectation(
        circ,
        state,
        values,
        obs,
        DiffMode.AD,
        n_shots=100000,
    )
    assert torch.allclose(exp_ad, exp_ad_sampled, atol=1e-01)

    # test density mat
    dm = density_mat(state)
    exp_ad_dm = expectation(circ, dm, values, obs, DiffMode.AD)
    exp_ad_sampled_dm = expectation(
        circ,
        dm,
        values,
        obs,
        DiffMode.AD,
        n_shots=100000,
    )
    assert torch.allclose(exp_ad_dm, exp_ad, atol=ATOL)
    assert torch.allclose(exp_ad, exp_ad_sampled_dm, atol=1e-01)


@pytest.mark.skip  # Adjoint Scale is currently not supported
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("n_qubits", [2])
def test_adjoint_scale(dtype: torch.dtype, batch_size: int, n_qubits: int) -> None:
    ops = [pyq.Scale(pyq.X(0), "theta_4")]

    theta_4_value = torch.rand(1, dtype=dtype)
    circ = pyq.QuantumCircuit(n_qubits, ops).to(dtype)

    state = pyq.random_state(n_qubits, batch_size, dtype=dtype)

    theta_4_ad = torch.tensor([theta_4_value], requires_grad=True)
    theta_4_adjoint = torch.tensor([theta_4_value], requires_grad=True)

    values_ad = torch.nn.ParameterDict(
        {
            "theta_4": theta_4_ad,
        }
    ).to(COMPLEX_TO_REAL_DTYPES[dtype])
    values_adjoint = torch.nn.ParameterDict(
        {
            "theta_4": theta_4_adjoint,
        }
    ).to(COMPLEX_TO_REAL_DTYPES[dtype])

    obs = pyq.Observable(pyq.Z(0)).to(dtype)
    exp_ad = expectation(circ, state, values_ad, obs, DiffMode.AD)
    exp_adjoint = expectation(circ, state, values_adjoint, obs, DiffMode.ADJOINT)

    grad_ad = torch.autograd.grad(
        exp_ad, tuple(values_ad.values()), torch.ones_like(exp_ad)
    )

    grad_adjoint = torch.autograd.grad(
        exp_adjoint, tuple(values_adjoint.values()), torch.ones_like(exp_adjoint)
    )

    assert len(grad_ad) == len(grad_adjoint)
    for i in range(len(grad_ad)):
        assert torch.allclose(grad_ad[i], grad_adjoint[i], atol=GRADCHECK_ATOL)


# Note pyq does not support using multiple times the same angle
@pytest.mark.parametrize("n_qubits", [3, 4, 5])
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("op_obs", [pyq.Z, pyq.X, pyq.Y])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("different_names", [False, True])
def test_all_diff_singlegap(
    n_qubits: int,
    batch_size: int,
    op_obs: Primitive,
    dtype: torch.dtype,
    different_names: bool,
) -> None:
    name_angles = "theta"
    dtype_val = COMPLEX_TO_REAL_DTYPES[dtype]

    if different_names:
        ops_rx = pyq.Sequence(
            [
                pyq.RX(i, param_name=name_angles + "_x_" + str(i))
                for i in range(n_qubits)
            ]
        )
        ops_rz = pyq.Sequence(
            [
                pyq.RZ(i, param_name=name_angles + "_z_" + str(i))
                for i in range(n_qubits)
            ]
        )
        values = {
            name_angles
            + "_x_"
            + str(i): torch.rand(1, dtype=dtype_val, requires_grad=True)
            for i in range(n_qubits)
        }
        values.update(
            {
                name_angles
                + "_z_"
                + str(i): torch.rand(1, dtype=dtype_val, requires_grad=True)
                for i in range(n_qubits)
            }
        )
    else:
        ops_rx = pyq.Sequence(
            [pyq.RX(i, param_name=name_angles + "_x") for i in range(n_qubits)]
        )
        ops_rz = pyq.Sequence(
            [pyq.RZ(i, param_name=name_angles + "_z") for i in range(n_qubits)]
        )
        values = {
            name_angles + "_x": torch.rand(1, dtype=dtype_val, requires_grad=True),
            name_angles + "_z": torch.rand(1, dtype=dtype_val, requires_grad=True),
        }
    cnot = pyq.CNOT(1, 2)
    ops = [ops_rx, ops_rz, cnot]

    circ = pyq.QuantumCircuit(n_qubits, ops).to(dtype)
    obs = pyq.Observable([op_obs(i) for i in range(n_qubits)]).to(dtype)
    state = pyq.random_state(n_qubits, batch_size, dtype=dtype)

    exp_ad = expectation(circ, state, values, obs, DiffMode.AD)
    exp_adjoint = expectation(circ, state, values, obs, DiffMode.ADJOINT)
    exp_gpsr = expectation(circ, state, values, obs, DiffMode.GPSR)

    assert torch.allclose(exp_ad, exp_adjoint)
    assert torch.allclose(exp_ad, exp_gpsr)

    grad_ad = torch.autograd.grad(
        exp_ad, tuple(values.values()), torch.ones_like(exp_ad), create_graph=True
    )

    grad_adjoint = torch.autograd.grad(
        exp_adjoint,
        tuple(values.values()),
        torch.ones_like(exp_adjoint),
        create_graph=True,
    )

    grad_gpsr = torch.autograd.grad(
        exp_gpsr, tuple(values.values()), torch.ones_like(exp_gpsr), create_graph=True
    )

    # check first order gradients
    assert len(grad_ad) == len(grad_adjoint) == len(grad_gpsr)
    for i in range(len(grad_ad)):
        assert torch.allclose(
            grad_ad[i], grad_adjoint[i], atol=GRADCHECK_ATOL
        ) and torch.allclose(grad_ad[i], grad_gpsr[i], atol=GRADCHECK_ATOL)

    # TODO higher order adjoint is not yet supported.
    # gradgrad_adjoint = torch.autograd.grad(
    #     grad_adjoint, tuple(values.values()), torch.ones_like(grad_adjoint)
    # )

    for i in range(len(grad_ad)):
        gradgrad_ad = torch.autograd.grad(
            grad_ad[i],
            tuple(values.values()),
            torch.ones_like(grad_ad[i]),
            create_graph=True,
        )

        gradgrad_gpsr = torch.autograd.grad(
            grad_gpsr[i],
            tuple(values.values()),
            torch.ones_like(grad_gpsr[i]),
            create_graph=True,
        )

        assert len(gradgrad_ad) == len(gradgrad_gpsr)

        # check second order gradients
        for j in range(len(gradgrad_ad)):
            assert torch.allclose(gradgrad_ad[j], gradgrad_gpsr[j], atol=GRADCHECK_ATOL)


def get_expval_grad(call_name, gate):
    x = torch.rand(1, requires_grad=True)
    fn = pyq.ConcretizedCallable(
        call_name=call_name, abstract_args=["x"], engine_name="torch"
    )
    circ = pyq.QuantumCircuit(1, [gate(0, fn)])
    state = pyq.zero_state(1)
    obs = pyq.Observable([pyq.Z(0)])
    values = {"x": x}

    expval = pyq.expectation(
        circuit=circ,
        state=state,
        values=values,
        observable=obs,
        diff_mode=pyq.DiffMode.ADJOINT,
    )

    grad = torch.autograd.grad(
        expval,
        tuple(values.values()),
        torch.ones_like(expval),
        retain_graph=True,
        allow_unused=True,
    )
    return grad, expval


@pytest.mark.parametrize("call_name", ["sin", "cos"])
@pytest.mark.parametrize("gate", [pyq.RX, pyq.RY, pyq.RZ])
def test_concretized_callable_differentiation(call_name, gate):
    x = torch.rand(1, requires_grad=True)
    fn = pyq.ConcretizedCallable(
        call_name=call_name, abstract_args=["x"], engine_name="torch"
    )
    circ = pyq.QuantumCircuit(1, [gate(0, fn)])
    state = pyq.zero_state(1)
    obs = pyq.Observable([pyq.Z(0)])
    values = {"x": x}
    expval = pyq.expectation(
        circuit=circ,
        state=state,
        values=values,
        observable=obs,
        diff_mode=pyq.DiffMode.ADJOINT,
    )

    grad = torch.autograd.grad(
        expval,
        tuple(values.values()),
        torch.ones_like(expval),
        retain_graph=True,
        allow_unused=True,
    )

    values = {"x": x, str(call_name): fn({"x": x})}
    _circ = pyq.QuantumCircuit(1, [gate(0, str(call_name))])

    _expval = pyq.expectation(
        circuit=_circ,
        state=state,
        values=values,
        observable=obs,
        diff_mode=pyq.DiffMode.ADJOINT,
    )

    _grad = torch.autograd.grad(
        _expval,
        tuple(values.values()),
        torch.ones_like(_expval),
        retain_graph=True,
        allow_unused=True,
    )
    assert expval == _expval
    assert torch.allclose(_grad[0], grad[0])
