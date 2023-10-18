from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor

import pyqtorch.modules as pyq

from .diff_rules import diff_rules


def param_dict(keys: Sequence[str], values: Sequence[Tensor]) -> dict[str, Tensor]:
    return {key: val for key, val in zip(keys, values)}


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
        return circuit(state, thetas, observable)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple:
        values = param_dict(ctx.param_keys, ctx.saved_tensors)
        AdjointExpectation.phi = AdjointExpectation.circuit(values)
        grads = []
        for op in reversed(ctx.circuit.operations):
            adj_mat = op.matrices(values).adjoint()
            AdjointExpectation.lmda = op.apply(adj_mat, AdjointExpectation.phi)  ## undo gate
            AdjointExpectation.mu = AdjointExpectation.lmda.clone()
            AdjointExpectation.mu = diff_rules[op.name](values[op.param_name])(
                AdjointExpectation.mu
            )
            grad = (
                2
                * AdjointExpectation.R
                * pyq.overlap(AdjointExpectation.lmda, AdjointExpectation.mu)
            )
            grads.append(grad)

        return (None, None, None, *grads)
