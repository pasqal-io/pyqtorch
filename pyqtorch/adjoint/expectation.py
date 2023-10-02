from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn
from torch.autograd import Function

import pyqtorch.modules as pyq
from .diff_rules import diff_rules


def param_dict(keys: Sequence[str], values: Sequence[Tensor]) -> dict[str, Tensor]:
    return {key: val for key, val in zip(keys, values)}


class AdjointExpectation(Function):
    lmda: torch.Tensor
    phi: torch.Tensor
    mu: torch.Tensor
    circuit: pyq.QuantumCircuit
    R: torch.Tensor

    @staticmethod
    def forward(
        ctx: Any,
        state: Tensor,
        param_keys: Sequence[str],
        *param_values: Tensor,
    ) -> Tensor:
        ctx.param_keys = param_keys
        ctx.save_for_backward(*param_values)
        AdjointExpectation.lmda = AdjointExpectation.circuit(state, param_dict(param_keys, param_values))
        return pyq.overlap(state, AdjointExpectation.lmda)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple:
        
        values = param_dict(ctx.param_keys, ctx.saved_tensors)
        AdjointExpectation.phi = AdjointExpectation.circuit(values)
        grads = []
        for op in reversed(ctx.circuit.operations):
            adj_mat = op.matrices(values).adjoint()
            AdjointExpectation.lmda = op.apply(adj_mat, AdjointExpectation.phi) ## undo gate
            AdjointExpectation.mu = AdjointExpectation.lmda.clone()
            AdjointExpectation.mu = diff_rules[op.name](values[op.param_name])(AdjointExpectation.mu)
            grad = 2 * AdjointExpectation.R * pyq.overlap(AdjointExpectation.lmda, AdjointExpectation.mu)
            grads.append(grad)

        return (None, None, None, *grads)
