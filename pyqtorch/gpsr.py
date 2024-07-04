from __future__ import annotations

from logging import getLogger
from typing import Any, Tuple

import torch
from torch import Tensor
from torch.autograd import Function

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
    def forward(
        ctx: Any,
        circuit: QuantumCircuit,
        state: Tensor,
        observable: Observable,
        param_names: list[str],
        *param_values: Tensor,
    ) -> Tensor:
        """The PSRExpectation forward call."""
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
        """The PSRExpectation Backward call."""
        values = param_dict(ctx.param_names, ctx.saved_tensors)
        shift = torch.tensor(torch.pi) / 2.0

        def expectation_fn(values: dict[str, Tensor]) -> torch.Tensor:
            """Use the PSRExpectation for nested grad calls."""
            return PSRExpectation.apply(
                ctx.circuit, ctx.state, ctx.observable, values.keys(), *values.values()
            )

        def single_gap_shift(
            param_name: str,
            values: dict[str, torch.Tensor],
            spectral_gap: torch.Tensor,
            shift: torch.Tensor = torch.tensor(torch.pi) / 2.0,
        ) -> torch.Tensor:
            """Describe single-gap GPSR."""
            shifted_values = values.copy()
            shifted_values[param_name] = shifted_values[param_name] + shift
            f_plus = expectation_fn(shifted_values)
            shifted_values = values.copy()
            shifted_values[param_name] = shifted_values[param_name] - shift
            f_min = expectation_fn(shifted_values)
            return (
                spectral_gap
                * (f_plus - f_min)
                / (4 * torch.sin(spectral_gap * shift / 2))
            )

        def multi_gap_shift(*args, **kwargs) -> torch.Tensor:
            """Describe multi_gap GPSR."""
            raise NotImplementedError("To be added,")

        def vjp(operation: Parametric, values: dict[str, torch.Tensor]) -> torch.Tensor:
            """Vector-jacobian product between `grad_out` and jacobians of parameters."""
            psr_fn = (
                multi_gap_shift if len(operation.spectral_gap) > 1 else single_gap_shift
            )
            return grad_out * psr_fn(  # type: ignore[operator]
                operation.param_name, values, operation.spectral_gap, shift
            )

        grads = []
        for op in ctx.circuit.flatten():
            if isinstance(op, Parametric) and values[op.param_name].requires_grad:  # type: ignore[index]
                grads.append(vjp(op, values))

        return (None, None, None, None, *grads)
