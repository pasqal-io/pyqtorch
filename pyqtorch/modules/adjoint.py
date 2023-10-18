from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor

import pyqtorch.modules as pyq
from pyqtorch.modules.parametric import RotationGate


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
        out_state = circuit(state, values)
        projected_state = observable(out_state, values)
        print(param_values)
        ctx.save_for_backward(torch.cat(param_values), out_state, projected_state)
        return pyq.overlap(out_state, projected_state)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple:
        param_values, state, projected_state = ctx.saved_tensors
        values = param_dict(ctx.param_names, param_values)
        grads = []
        for op in ctx.circuit.reverse().operations:
            state = op.apply_dagger(state, values)
            if isinstance(op, RotationGate):
                mu = op.apply_jacobian(state, values)
                grads.append(grad_out * 2 * pyq.overlap(projected_state, mu))
            else:
                grads.append(None)
            projected_state = op.apply_dagger(projected_state, values)
        return (None, None, None, None, *list(reversed(grads)))
