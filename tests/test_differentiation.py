from __future__ import annotations

import pytest
import torch

import pyqtorch as pyq
from pyqtorch import DiffMode, expectation
from pyqtorch.matrices import COMPLEX_TO_REAL_DTYPES
from pyqtorch.parametric import Parametric
from pyqtorch.utils import (
    GRADCHECK_ATOL,
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
    obs = pyq.QuantumCircuit(n_qubits, [pyq.Z(0)])

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

    values_ad = {"theta_0": theta_0_ad, "theta_1": theta_1_ad, "theta_2": theta_2_ad}
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


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("n_qubits", [3, 4])
def test_differentiate_circuit(
    dtype: torch.dtype, batch_size: int, n_qubits: int
) -> None:
    ops = [
        pyq.Y(1),
        pyq.RX(0, "theta_0"),
        pyq.PHASE(0, "theta_1"),
        pyq.CSWAP(0, (1, 2)),
        pyq.CRX(1, 2, "theta_2"),
        pyq.CPHASE(1, 2, "theta_3"),
        pyq.CNOT(0, 1),
        pyq.Toffoli((2, 1), 0),
    ]
    circ = pyq.QuantumCircuit(n_qubits, ops).to(dtype)
    all_param_names = [
        op.param_name
        for op in circ.flatten()
        if isinstance(op, Parametric) and isinstance(op.param_name, str)
    ]
    theta_vals = [torch.rand(1, dtype=dtype) for p in all_param_names]

    state = pyq.random_state(n_qubits, batch_size, dtype=dtype)

    theta_ad = [torch.tensor([t], requires_grad=True) for t in theta_vals]
    theta_adjoint = [torch.tensor([t], requires_grad=True) for t in theta_vals]
    theta_gpsr = [torch.tensor([t], requires_grad=True) for t in theta_vals]

    values_ad = torch.nn.ParameterDict(
        {t: tval for (t, tval) in zip(all_param_names, theta_ad)}
    ).to(COMPLEX_TO_REAL_DTYPES[dtype])
    values_adjoint = torch.nn.ParameterDict(
        {t: tval for (t, tval) in zip(all_param_names, theta_adjoint)}
    ).to(COMPLEX_TO_REAL_DTYPES[dtype])
    values_gpsr = torch.nn.ParameterDict(
        {t: tval for (t, tval) in zip(all_param_names, theta_gpsr)}
    ).to(COMPLEX_TO_REAL_DTYPES[dtype])

    obs = pyq.QuantumCircuit(n_qubits, [pyq.Z(0)]).to(dtype)
    exp_ad = expectation(circ, state, values_ad, obs, DiffMode.AD)
    exp_adjoint = expectation(circ, state, values_adjoint, obs, DiffMode.ADJOINT)
    exp_gpsr = expectation(circ, state, values_gpsr, obs, DiffMode.GPSR)

    grad_ad = torch.autograd.grad(
        exp_ad, tuple(values_ad.values()), torch.ones_like(exp_ad), create_graph=True
    )[0]

    grad_adjoint = torch.autograd.grad(
        exp_adjoint,
        tuple(values_adjoint.values()),
        torch.ones_like(exp_adjoint),
        create_graph=True,
    )[0]

    grad_gpsr = torch.autograd.grad(
        exp_gpsr,
        tuple(values_gpsr.values()),
        torch.ones_like(exp_gpsr),
        create_graph=True,
    )[0]

    assert len(grad_ad) == len(grad_adjoint) == len(grad_gpsr)
    for i in range(len(grad_ad)):
        assert torch.allclose(grad_ad[i], grad_adjoint[i], atol=GRADCHECK_ATOL)
        assert torch.allclose(grad_ad[i], grad_gpsr[i], atol=GRADCHECK_ATOL)

    # gradgrad_ad = torch.autograd.grad(
    #     grad_ad, tuple(values_ad.values()), torch.ones_like(grad_ad), create_graph=True
    # )[0]

    # TODO higher order adjoint is not yet supported.
    # gradgrad_adjoint = torch.autograd.grad(
    #     grad_adjoint, tuple(values_adjoint.values()), torch.ones_like(grad_adjoint)
    # )

    # gradgrad_gpsr = torch.autograd.grad(
    #     grad_gpsr,
    #     tuple(values_gpsr.values()),
    #     torch.ones_like(grad_gpsr),
    #     create_graph=True,
    # )[0]

    # assert len(gradgrad_ad) == len(gradgrad_gpsr)

    # # check second order gradients
    # for i in range(len(gradgrad_ad)):
    #     assert torch.allclose(gradgrad_ad[i], gradgrad_gpsr[i], atol=GRADCHECK_ATOL)


@pytest.mark.xfail  # investigate
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

    obs = pyq.QuantumCircuit(n_qubits, [pyq.Z(0)]).to(dtype)
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
