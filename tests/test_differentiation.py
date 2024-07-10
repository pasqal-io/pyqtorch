from __future__ import annotations

import pytest
import torch

import pyqtorch as pyq
from pyqtorch import DiffMode, MeasurementProtocols, expectation
from pyqtorch.matrices import COMPLEX_TO_REAL_DTYPES
from pyqtorch.utils import (
    GRADCHECK_ATOL,
)


# TODO add GPSR when multigap is implemented for this test
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
    theta_0_value = torch.rand(1, dtype=dtype)
    theta_1_value = torch.rand(1, dtype=dtype)
    theta_2_value = torch.rand(1, dtype=dtype)
    theta_3_value = torch.rand(1, dtype=dtype)
    circ = pyq.QuantumCircuit(n_qubits, ops).to(dtype)
    state = pyq.random_state(n_qubits, batch_size, dtype=dtype)

    theta_0_ad = torch.tensor([theta_0_value], requires_grad=True)
    theta_0_adjoint = torch.tensor([theta_0_value], requires_grad=True)

    theta_1_ad = torch.tensor([theta_1_value], requires_grad=True)
    theta_1_adjoint = torch.tensor([theta_1_value], requires_grad=True)

    theta_2_ad = torch.tensor([theta_2_value], requires_grad=True)
    theta_2_adjoint = torch.tensor([theta_2_value], requires_grad=True)

    theta_3_ad = torch.tensor([theta_3_value], requires_grad=True)
    theta_3_adjoint = torch.tensor([theta_3_value], requires_grad=True)

    values_ad = torch.nn.ParameterDict(
        {
            "theta_0": theta_0_ad,
            "theta_1": theta_1_ad,
            "theta_2": theta_2_ad,
            "theta_3": theta_3_ad,
        }
    ).to(COMPLEX_TO_REAL_DTYPES[dtype])
    values_adjoint = torch.nn.ParameterDict(
        {
            "theta_0": theta_0_adjoint,
            "theta_1": theta_1_adjoint,
            "theta_2": theta_2_adjoint,
            "theta_3": theta_3_adjoint,
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


# Note pyq does not support using multiple times the same angle
@pytest.mark.parametrize("n_qubits", [3, 4, 5])
def test_all_diff_singlegap(n_qubits: int) -> None:
    name_angles = "theta"

    ops_rx = pyq.Sequence(
        [pyq.RX(i, param_name=name_angles + "_x_" + str(i)) for i in range(n_qubits)]
    )
    ops_rz = pyq.Sequence(
        [pyq.RZ(i, param_name=name_angles + "_z_" + str(i)) for i in range(n_qubits)]
    )
    cnot = pyq.CNOT(1, 2)
    ops = [ops_rx, ops_rz, cnot]

    circ = pyq.QuantumCircuit(n_qubits, ops)
    obs = pyq.QuantumCircuit(n_qubits, [pyq.Z(0)])
    state = pyq.random_state(n_qubits)

    values = {
        name_angles + "_x_" + str(i): torch.rand(1, requires_grad=True)
        for i in range(n_qubits)
    }
    values.update(
        {
            name_angles + "_z_" + str(i): torch.rand(1, requires_grad=True)
            for i in range(n_qubits)
        }
    )

    exp_ad = expectation(circ, state, values, obs, DiffMode.AD)
    exp_adjoint = expectation(circ, state, values, obs, DiffMode.ADJOINT)
    exp_gpsr = expectation(circ, state, values, obs, DiffMode.GPSR)

    assert torch.allclose(exp_ad, exp_adjoint)
    assert torch.allclose(exp_ad, exp_gpsr)

    grad_ad = torch.autograd.grad(
        exp_ad, tuple(values.values()), torch.ones_like(exp_ad), create_graph=True
    )[0]

    grad_adjoint = torch.autograd.grad(
        exp_adjoint,
        tuple(values.values()),
        torch.ones_like(exp_adjoint),
        create_graph=True,
    )[0]

    grad_gpsr = torch.autograd.grad(
        exp_gpsr, tuple(values.values()), torch.ones_like(exp_gpsr), create_graph=True
    )[0]

    gradgrad_ad = torch.autograd.grad(
        grad_ad, tuple(values.values()), torch.ones_like(grad_ad)
    )

    # TODO higher order adjoint is not yet supported.
    # gradgrad_adjoint = torch.autograd.grad(
    #     grad_adjoint, tuple(values.values()), torch.ones_like(grad_adjoint)
    # )

    gradgrad_gpsr = torch.autograd.grad(
        grad_gpsr, tuple(values.values()), torch.ones_like(grad_gpsr)
    )

    # check first order gradients
    assert len(grad_ad) == len(grad_adjoint) == len(grad_gpsr)
    for i in range(len(grad_ad)):
        assert torch.allclose(
            grad_ad[i], grad_adjoint[i], atol=GRADCHECK_ATOL
        ) and torch.allclose(grad_ad[i], grad_gpsr[i], atol=GRADCHECK_ATOL)

    assert len(gradgrad_ad) == len(gradgrad_gpsr)

    # check second order gradients
    for i in range(len(gradgrad_ad)):
        assert torch.allclose(gradgrad_ad[i], gradgrad_gpsr[i], atol=GRADCHECK_ATOL)


@pytest.mark.parametrize("gate_type", ["scale", "hamevo", ""])
def test_compatibility_gpsr(gate_type: str) -> None:

    pname = "theta_0"
    if gate_type == "scale":
        seq_gate = pyq.Sequence([pyq.X(0)])
        scale = pyq.Scale(seq_gate, pname)
        ops = [scale]
    elif gate_type == "hamevo":
        hamevo = pyq.HamiltonianEvolution(pyq.Sequence([pyq.X(0)]), pname, (0,))
        ops = [hamevo]
    else:
        ops = [pyq.RY(0, pname), pyq.RZ(0, pname)]

    circ = pyq.QuantumCircuit(1, ops)
    obs = pyq.QuantumCircuit(1, [pyq.Z(0)])
    state = pyq.zero_state(1)

    param_value = torch.pi / 2
    values = {"theta_0": torch.tensor([param_value], requires_grad=True)}
    with pytest.raises(ValueError):
        exp_gpsr = expectation(circ, state, values, obs, DiffMode.GPSR)

        grad_gpsr = torch.autograd.grad(
            exp_gpsr, tuple(values.values()), torch.ones_like(exp_gpsr)
        )


@pytest.mark.parametrize("n_qubits", [3, 4, 5])
def test_all_diff_singlegap(n_qubits: int) -> None:
    name_angles = "theta"

    ops_rx = pyq.Sequence(
        [pyq.RX(i, param_name=name_angles + "_x_" + str(i)) for i in range(n_qubits)]
    )
    ops_rz = pyq.Sequence(
        [pyq.RZ(i, param_name=name_angles + "_z_" + str(i)) for i in range(n_qubits)]
    )
    cnot = pyq.CNOT(1, 2)
    ops = [ops_rx, ops_rz, cnot]

    circ = pyq.QuantumCircuit(n_qubits, ops)
    obs = pyq.QuantumCircuit(n_qubits, [pyq.Z(0)])
    state = pyq.random_state(n_qubits)

    values = {
        name_angles + "_x_" + str(i): torch.rand(1, requires_grad=True)
        for i in range(n_qubits)
    }
    values.update(
        {
            name_angles + "_z_" + str(i): torch.rand(1, requires_grad=True)
            for i in range(n_qubits)
        }
    )

    exp_gpsr = expectation(circ, state, values, obs, DiffMode.GPSR)

    tomo_protocol = MeasurementProtocols("tomography", {"n_shots": 100000})
    exp_gpsr_shots = expectation(
        circ, state, values, obs, DiffMode.GPSR, measurement=tomo_protocol
    )
    assert torch.allclose(exp_gpsr, exp_gpsr_shots, atol=1.0e-2)
