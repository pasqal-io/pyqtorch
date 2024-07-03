from __future__ import annotations

from logging import getLogger
from typing import Any, Tuple

import torch
from torch import Tensor, no_grad
from torch.autograd import Function

import pyqtorch as pyq
from pyqtorch.analog import Observable
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.parametric import Parametric
from pyqtorch.utils import inner_prod, param_dict

logger = getLogger(__name__)


class PSRExpectation(Function):
    """
    Describe PSR
    """

    @staticmethod
    @no_grad()
    def forward(
        ctx: Any,
        circuit: QuantumCircuit,
        observable: Observable,
        state: Tensor,
        param_names: list[str],
        *param_values: Tensor,
    ) -> Tensor:
        ctx.circuit = circuit
        ctx.observable = observable
        ctx.param_names = param_names
        ctx.state = state
        values = param_dict(param_names, param_values)
        ctx.out_state = circuit.run(state, values)
        ctx.projected_state = observable.run(ctx.out_state, values)
        ctx.save_for_backward(*param_values)
        return inner_prod(ctx.out_state, ctx.projected_state).real

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> Tuple[None, ...]:
        param_values = ctx.saved_tensors
        values = param_dict(ctx.param_names, param_values)
        grads_dict = {k: None for k in values.keys()}
        shift = torch.tensor(torch.pi) / 2.0

        for op in ctx.circuit.flatten():
            if isinstance(op, Parametric) and isinstance(op.param_name, str):
                spectrum = torch.linalg.eigvalsh(op.pauli).reshape(-1, 1)
                spectral_gap = torch.unique(
                    torch.abs(torch.tril(spectrum - spectrum.T))
                )
                spectral_gap = spectral_gap[spectral_gap.nonzero()]
                assert (
                    len(spectral_gap) == 1
                ), "PSRExpectation only works on single_gap for now."

                if values[op.param_name].requires_grad:
                    with no_grad():
                        copied_values = values.copy()
                        copied_values[op.param_name] += shift
                        f_plus = pyq.expectation(
                            ctx.circuit, ctx.state, copied_values, ctx.observable
                        )
                        copied_values[op.param_name] -= 2.0 * shift
                        f_min = pyq.expectation(
                            ctx.circuit, ctx.state, copied_values, ctx.observable
                        )

                    grad = (
                        spectral_gap
                        * (f_plus - f_min)
                        / (4 * torch.sin(spectral_gap * shift / 2))
                    )
                    grad *= grad_out
                if grads_dict[op.param_name] is not None:
                    grads_dict[op.param_name] += grad
                else:
                    grads_dict[op.param_name] = grad
            else:
                logger.error(f"PSRExpectation does not support operation: {type(op)}.")
        return (None, None, None, None, *grads_dict.values())
