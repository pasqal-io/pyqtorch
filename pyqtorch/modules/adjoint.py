from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor

import pyqtorch as pyq
from pyqtorch.modules.parametric import Parametric


def param_dict(keys: Sequence[str], values: Sequence[Tensor]) -> dict[str, Tensor]:
    return {key: val for key, val in zip(keys, values)}


class AdjointExpectation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        circuit: pyq.QuantumCircuit,
        observable: pyq.QuantumCircuit,
        state: Tensor,
        param_names: list[str],
        *param_values: Tensor,
    ) -> Tensor:
        ctx.circuit = circuit
        ctx.observable = observable
        ctx.param_names = param_names
        values = param_dict(param_names, param_values)
        ctx.out_state = circuit.run(state, values)
        ctx.projected_state = observable.run(ctx.out_state, values)
        ctx.save_for_backward(*param_values)
        return pyq.overlap(ctx.out_state, ctx.projected_state)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple:
        param_values = ctx.saved_tensors
        values = param_dict(ctx.param_names, param_values)
        grads: list = []
        for op in ctx.circuit.dagger().operations:
            ctx.out_state = op.apply_dagger(ctx.out_state, values)
            if isinstance(op, Parametric):
                mu = op.apply_jacobian(ctx.out_state, values)
                grads = [grad_out * 2 * pyq.overlap(ctx.projected_state, mu)] + grads
            else:
                grads = [None] + grads
            ctx.projected_state = op.apply_dagger(ctx.projected_state, values)
        return (None, None, None, None, *grads)
