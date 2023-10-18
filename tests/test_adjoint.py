from __future__ import annotations

import torch

import pyqtorch.modules as pyq
from pyqtorch.modules.circuit import DiffMode


def test_adjoint_diff() -> None:
    gate = pyq.RX([0], 1)
    adjoint_circ = pyq.QuantumCircuit(1, [gate], DiffMode.ADJOINT)
    ad_circ = pyq.QuantumCircuit(1, [gate], DiffMode.AD)

    param_value = torch.pi / 2

    state = pyq.zero_state(1)
    thetas_ad = torch.tensor([param_value], requires_grad=True)
    thetas_adjoint = torch.tensor([param_value], requires_grad=True)
    obs = pyq.QuantumCircuit(1, [pyq.Z([0], 1)])
    exp_ad = ad_circ.expectation(state, thetas_ad, obs)
    exp_adjoint = adjoint_circ.expectation(state, thetas_adjoint, obs)

    grad_ad = torch.autograd.grad(exp_ad, thetas_ad, torch.ones_like(exp_ad))
    grad_adjoint = torch.autograd.grad(exp_adjoint, thetas_adjoint, torch.ones_like(exp_adjoint))

    assert torch.allclose(grad_ad[0], grad_adjoint[0])
