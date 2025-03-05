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


@pytest.mark.parametrize("n_qubits", [3, 4, 5])
@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_adjoint_diff(n_qubits: int, n_layers: int) -> None:
    rx = pyq.RX(0, param_name="theta_0")
    cry = pyq.CPHASE(0, 1, param_name="theta_1")
    rz = pyq.RZ(2, param_name="theta_2")

    cnot = pyq.CNOT(1, 2)
    ops = [rx, cry, rz, cnot] * n_layers
    circ = pyq.QuantumCircuit(n_qubits, ops)
    obs = pyq.Observable(pyq.Z(0))

    theta_0_value = torch.pi / 2
    theta_1_value = torch.pi
    theta_2_value = torch.pi / 4

    state = pyq.zero_state(n_qubits)

    theta_0_ad = torch.tensor([theta_0_value], requires_grad=True)
    thetas_0_adjoint = torch.tensor([theta_0_value], requires_grad=True)

    theta_1_ad = torch.tensor([theta_1_value], requires_grad=True)
    thetas_1_adjoint = torch.tensor([theta_1_value], requires_grad=True)

    theta_2_ad = torch.tensor([theta_2_value], requires_grad=True)
    thetas_2_adjoint = torch.tensor([theta_2_value], requires_grad=True)

    values_ad = {
        "theta_0": theta_0_ad,
        "theta_1": theta_1_ad,
        "theta_2": theta_2_ad,
    }
    values_adjoint = {
        "theta_0": thetas_0_adjoint,
        "theta_1": thetas_1_adjoint,
        "theta_2": thetas_2_adjoint,
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
        assert torch.allclose(grad_ad[i], grad_adjoint[i], atol=GRADCHECK_ATOL)

    # TODO higher order adjoint is not yet supported.
    # gradgrad_adjoint = torch.autograd.grad(
    #     grad_adjoint, tuple(values_adjoint.values()), torch.ones_like(grad_adjoint)
    # )


@pytest.mark.parametrize("n_qubits", [3, 4, 5])
@pytest.mark.parametrize("n_layers", [1, 2, 3])
@pytest.mark.parametrize("diagonal", [True, False])
def test_adjoint_diff_hamevo(n_qubits: int, n_layers: int, diagonal: bool) -> None:

    cnot = pyq.CNOT(1, 2)
    ham = random_pauli_hamiltonian(
        n_qubits, k_1q=n_qubits, k_2q=0, default_scale_coeffs=1.0, diagonal=diagonal
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


@pytest.mark.xfail  # Adjoint Scale is currently not supported
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
