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
        out_state = circuit(state, thetas)
        projected_state = observable(out_state, thetas)
        ctx.save_for_backward(thetas, out_state, projected_state)
        return pyq.overlap(out_state, projected_state)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple:
        thetas, state, projected_state = ctx.saved_tensors
        grads = []
        for op in ctx.circuit.reverse().operations:
            state = op.apply_dagger(state, thetas)
            if isinstance(op, RotationGate):
                mu = op.apply_jacobian(state, thetas)
                grads.append(grad_out * 2 * pyq.overlap(projected_state, mu))
            else:
                grads.append(None)
            projected_state = op.apply_dagger(projected_state, thetas)
        return (None, None, None, *reversed(grads))
