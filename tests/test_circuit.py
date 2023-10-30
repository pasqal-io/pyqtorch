from __future__ import annotations

import pytest
import torch

import pyqtorch as pyq
from pyqtorch.circuit import DiffMode


def test_adjoint_diff() -> None:
    rx = pyq.RX(0, param_name="theta_0")
    cry = pyq.CPHASE(0, 1, param_name="theta_1")
    rz = pyq.RZ(2, param_name="theta_2")
    cnot = pyq.CNOT(1, 2)
    ops = [rx, cry, rz, cnot]
    n_qubits = 3
    adjoint_circ = pyq.QuantumCircuit(n_qubits, ops, DiffMode.ADJOINT)
    ad_circ = pyq.QuantumCircuit(n_qubits, ops, DiffMode.AD)
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
    exp_ad = ad_circ.expectation(values_ad, obs, state)
    exp_adjoint = adjoint_circ.expectation(values_adjoint, obs, state)

    grad_ad = torch.autograd.grad(exp_ad, tuple(values_ad.values()), torch.ones_like(exp_ad))

    grad_adjoint = torch.autograd.grad(
        exp_adjoint, tuple(values_adjoint.values()), torch.ones_like(exp_adjoint)
    )

    assert len(grad_ad) == len(grad_adjoint)
    for i in range(len(grad_ad)):
        assert torch.allclose(grad_ad[i], grad_adjoint[i])


@pytest.mark.parametrize("diff_mode", [DiffMode.AD, DiffMode.ADJOINT])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("n_qubits", [3, 4])
def test_differentiate_circuit(diff_mode: DiffMode, batch_size: int, n_qubits: int) -> None:
    ops = [
        pyq.RX(0, "phi"),
        pyq.PHASE(0, "theta"),
        pyq.CSWAP((0, 1), 2),
        pyq.CPHASE(1, 2, "epsilon"),
        pyq.CNOT(0, 1),
        pyq.Toffoli((2, 1), 0),
    ]
    circ = pyq.QuantumCircuit(n_qubits, ops, diff_mode=diff_mode)
    state = pyq.random_state(n_qubits, batch_size)
    phi = torch.rand(batch_size, requires_grad=True)
    theta = torch.rand(batch_size, requires_grad=True)
    epsilon = torch.rand(batch_size, requires_grad=True)
    values = {"phi": phi, "theta": theta, "epsilon": epsilon}
    assert circ(state, values).size() == tuple(2 for _ in range(n_qubits)) + (batch_size,)
    state = pyq.random_state(n_qubits, batch_size=batch_size)

    def _fwd(phi: torch.Tensor, theta: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        return circ(state, {"phi": phi, "theta": theta, "epsilon": epsilon})

    assert torch.autograd.gradcheck(_fwd, (phi, theta, epsilon))
