from __future__ import annotations

import torch

import pyqtorch.modules as pyq
from pyqtorch.modules.circuit import DiffMode


def test_adjoint_diff() -> None:
    rx = pyq.RX([0], 2, param_name="theta")
    ry = pyq.RY([1], 2, param_name="epsilon")
    ops = [rx, ry]
    n_qubits = 2
    adjoint_circ = pyq.QuantumCircuit(n_qubits, ops, DiffMode.ADJOINT)
    ad_circ = pyq.QuantumCircuit(n_qubits, ops, DiffMode.AD)
    obs = pyq.QuantumCircuit(n_qubits, [pyq.Z([0], n_qubits)])

    theta_value = torch.pi / 2
    epsilon_value = torch.pi / 4

    state = pyq.zero_state(n_qubits)

    thetas_ad = torch.tensor([theta_value], requires_grad=True)
    thetas_adjoint = torch.tensor([theta_value], requires_grad=True)

    epsilon_ad = torch.tensor([epsilon_value], requires_grad=True)
    epsilon_adjoint = torch.tensor([epsilon_value], requires_grad=True)

    exp_ad = ad_circ.expectation(state, {"theta": thetas_ad, "epsilon": epsilon_ad}, obs)
    exp_adjoint = adjoint_circ.expectation(
        state, {"theta": thetas_adjoint, "epsilon": epsilon_adjoint}, obs
    )

    grad_ad = torch.autograd.grad(exp_ad, (thetas_ad, epsilon_ad), torch.ones_like(exp_ad))
    grad_adjoint = torch.autograd.grad(
        exp_adjoint, (thetas_adjoint, epsilon_adjoint), torch.ones_like(exp_adjoint)
    )

    assert torch.allclose(grad_ad[0], grad_adjoint[0])
    assert torch.allclose(grad_ad[1], grad_adjoint[1])

    assert torch.autograd.gradcheck(lambda thetas: adjoint_circ(state, thetas), thetas_adjoint)
