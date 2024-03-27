from __future__ import annotations

from typing import Any

from torch import Tensor, no_grad, zeros
from torch.autograd import Function

from pyqtorch.apply import apply_operator
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.parametric import Parametric
from pyqtorch.utils import inner_prod, param_dict


class AdjointExpectation(Function):
    @staticmethod
    @no_grad()
    def forward(
        ctx: Any,
        circuit: QuantumCircuit,
        observable: QuantumCircuit,
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
        return inner_prod(ctx.out_state, ctx.projected_state).real

    @staticmethod
    @no_grad()
    def backward(ctx: Any, grad_out: Tensor) -> tuple:
        param_values = ctx.saved_tensors
        values = param_dict(ctx.param_names, param_values)
        grads_dict = values.copy()
        for op in ctx.circuit.reverse():
            ctx.out_state = apply_operator(ctx.out_state, op.dagger(values), op.qubit_support)
            if isinstance(op, Parametric):
                if values[op.param_name].requires_grad:
                    mu = apply_operator(ctx.out_state, op.jacobian(values), op.qubit_support)
                    grad = grad_out * 2 * inner_prod(ctx.projected_state, mu).real
                else:
                    grad = zeros(1)

                grads_dict[op.param_name] = grad

            ctx.projected_state = apply_operator(
                ctx.projected_state, op.dagger(values), op.qubit_support
            )
        return (None, None, None, None, *grads_dict.values())
