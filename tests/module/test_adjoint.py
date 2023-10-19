from __future__ import annotations

import torch

import pyqtorch as pyq
from pyqtorch.circuit import DiffMode


def test_adjoint_diff() -> None:
    rx = pyq.RX([0], 2, param_name="theta")
    ry = pyq.CRY([0, 1], 2, param_name="epsilon")
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

    assert len(grad_ad) == len(grad_adjoint)
    assert torch.allclose(grad_ad[0], grad_adjoint[0])
    assert torch.allclose(grad_ad[1], grad_adjoint[1])


def _adjoint_diff_multiparam() -> None:
    ops = []

    g0 = pyq.RX([0], 2, param_name="theta_0")
    g1 = pyq.RY([0], 2, param_name="theta_1")
    g2 = pyq.RX([0], 2, param_name="theta_2")

    g3 = pyq.RX([1], 2, param_name="theta_3")
    g4 = pyq.RY([1], 2, param_name="theta_4")
    g5 = pyq.RX([1], 2, param_name="theta_5")
    cnot = pyq.CNOT([0, 1], 2)
    ops = [g0, g1, g2, g3, g4, g5, cnot]
    n_qubits = 2
    adjoint_circ = pyq.QuantumCircuit(n_qubits, ops, DiffMode.ADJOINT)
    ad_circ = pyq.QuantumCircuit(n_qubits, ops, DiffMode.AD)
    obs = pyq.QuantumCircuit(n_qubits, [pyq.Z([0], n_qubits)])

    values = torch.rand(1)

    state = pyq.zero_state(n_qubits)

    thetas_ad = torch.tensor([values], requires_grad=True)
    thetas_adjoint = torch.tensor([values], requires_grad=True)

    epsilon_ad = torch.tensor([values], requires_grad=True)
    epsilon_adjoint = torch.tensor([values], requires_grad=True)

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
