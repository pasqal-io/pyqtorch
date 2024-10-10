from __future__ import annotations

from logging import getLogger
from typing import Any, Tuple

from torch import Tensor, no_grad
from torch.autograd import Function

from pyqtorch.apply import apply_operator
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.composite import Scale
from pyqtorch.embed import Embedding
from pyqtorch.hamiltonians import HamiltonianEvolution, Observable
from pyqtorch.primitives import Parametric, Primitive
from pyqtorch.utils import inner_prod, param_dict

logger = getLogger(__name__)


class AdjointExpectation(Function):
    """
    The adjoint differentiation method (https://arxiv.org/pdf/2009.02823.pdf) implemented as a
    custom torch.autograd.Function. It is able to perform a backward pass in O(P) time
    and maintaining atmost 3 states where P is the number of parameters in a variational circuit.

    Arguments:
        circuit: A QuantumCircuit instance
        observable: A hamiltonian.
        state: A state in the form of [2 * n_qubits + [batch_size]]
        param_names: A list of parameter names.
        *param_values: A unpacked tensor of values for each parameter.

    The forward method expects each of the above arguments, computes the expectation value
    and stores all of the above arguments in the 'ctx' (context) object along with the
    the state after applying 'circuit', 'out_state', and the 'projected_state', i.e. after applying
    'observable' to 'out_state'.

    In the 'backward', the circuit is traversed in reverse order,
    starting from the last gate to the first gate, 'undoing' each gate using the dagger operation on
    both 'out_state' and 'projected_state'.

    In case of a parametric gate (with a parameter which requires_grad):
    (1) We compute the jacobian of the operator and apply it to 'out_state'
    (2) Compute the inner product with 'projected_state' which yields the gradient.
    """

    @staticmethod
    @no_grad()
    def forward(
        ctx: Any,
        circuit: QuantumCircuit,
        state: Tensor,
        observable: Observable,
        embedding: Embedding | None,
        param_names: list[str],
        *param_values: Tensor,
    ) -> Tensor:
        ctx.circuit = circuit
        ctx.observable = observable
        ctx.embedding = embedding
        ctx.param_names = param_names
        values = param_dict(param_names, param_values)
        if embedding is not None:
            msg = "AdjointExpectation does not support Embedding."
            logger.error(msg)
            raise NotImplementedError(msg)
        ctx.out_state = circuit.run(state, values, embedding)
        ctx.projected_state = observable(ctx.out_state, values)
        ctx.save_for_backward(*param_values)
        return inner_prod(ctx.out_state, ctx.projected_state).real

    @staticmethod
    @no_grad()
    def backward(ctx: Any, grad_out: Tensor) -> Tuple[None, ...]:
        param_values = ctx.saved_tensors
        values = param_dict(ctx.param_names, param_values)
        grads_dict = {k: None for k in values.keys()}
        for op in ctx.circuit.flatten()[::-1]:
            if isinstance(op, (Primitive, Parametric, HamiltonianEvolution)):
                ctx.out_state = apply_operator(
                    ctx.out_state, op.dagger(values, ctx.embedding), op.qubit_support
                )
                if isinstance(op, (Parametric, HamiltonianEvolution)) and isinstance(
                    op.param_name, str
                ):
                    if values[op.param_name].requires_grad:
                        mu = apply_operator(
                            ctx.out_state,
                            op.jacobian(values, ctx.embedding),
                            op.qubit_support,
                        )
                        grad = grad_out * 2 * inner_prod(ctx.projected_state, mu).real
                    if grads_dict[op.param_name] is not None:
                        grads_dict[op.param_name] += grad
                    else:
                        grads_dict[op.param_name] = grad

                ctx.projected_state = apply_operator(
                    ctx.projected_state,
                    op.dagger(values, ctx.embedding),
                    op.qubit_support,
                )

            elif isinstance(op, Scale):
                # TODO: Properly fix or not support Scale in adjoint at all
                logger.error(
                    f"AdjointExpectation does not support operation: {type(op)}."
                )
                if not len(op.operations) == 1 and isinstance(
                    op.operations[0], Primitive
                ):
                    logger.error(
                        "Adjoint can only be used on Scale with Primitive blocks."
                    )
                ctx.out_state = apply_operator(
                    ctx.out_state, op.dagger(values, ctx.embedding), op.qubit_support
                )
                scaled_pyq_op = op.operations[0]
                if (
                    isinstance(scaled_pyq_op, Parametric)
                    and isinstance(scaled_pyq_op.param_name, str)
                    and values[scaled_pyq_op.param_name].requires_grad
                ):
                    mu = apply_operator(
                        ctx.out_state,
                        scaled_pyq_op.jacobian(values, ctx.embedding),
                        scaled_pyq_op.qubit_support,
                    )
                    grads_dict[scaled_pyq_op.param_name] = (
                        grad_out * 2 * inner_prod(ctx.projected_state, mu).real
                    )

                if values[op.param_name].requires_grad:  # type: ignore [index]
                    grads_dict[op.param_name] = grad_out * 2 * -values[op.param_name]  # type: ignore [index]
                ctx.projected_state = apply_operator(
                    ctx.projected_state,
                    op.dagger(values, ctx.embedding),
                    op.qubit_support,
                )
            else:
                logger.error(
                    f"AdjointExpectation does not support operation: {type(op)}."
                )
        return (None, None, None, None, None, *grads_dict.values())
