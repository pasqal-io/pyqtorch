from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

import pyqtorch.modules as pyq
from pyqtorch.modules.parametric import RotationGate


class AdjointExpectation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        circuit: pyq.QuantumCircuit,
        observable: pyq.QuantumCircuit,
        state: Tensor,
        thetas: Tensor,
    ) -> Tensor:
        ctx.circuit = circuit
        ctx.observable = observable
        ctx.thetas = thetas
        ctx.save_for_backward(thetas)
        out_state = circuit(state, thetas)
        projected_state = observable(out_state, thetas)
        ctx.save_for_backward(out_state)
        ctx.save_for_backward(projected_state)
        return pyq.overlap(out_state, projected_state)

    @staticmethod
    def backward(ctx: Any, state: torch.Tensor, grad_out: Tensor) -> tuple:
        thetas, state, projected_state = ctx.saved_tensors
        grads = []
        for op in ctx.circuit.reverse():
            state = op.apply_dagger(thetas, state)
            if isinstance(op, RotationGate):
                mu = op.apply_jacobian(thetas, state)
                grads.append(grad_out * 2 * pyq.overlap(projected_state, mu))
            projected_state = op.apply_dagger(thetas, projected_state)
        return (None, None, None, *grads)
