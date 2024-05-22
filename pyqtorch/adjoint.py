from __future__ import annotations

from typing import Any, Tuple

from torch import Tensor, no_grad
from torch.autograd import Function

from pyqtorch.analog import Hamiltonian
from pyqtorch.apply import apply_operator
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.parametric import Parametric
from pyqtorch.primitive import Primitive
from pyqtorch.utils import DiffMode, inner_prod, param_dict


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
        observable: Hamiltonian,
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
    def backward(ctx: Any, grad_out: Tensor) -> Tuple[None, ...]:
        param_values = ctx.saved_tensors
        values = param_dict(ctx.param_names, param_values)
        grads_dict = {k: None for k in values.keys()}
        for op in ctx.circuit.flatten()[::-1]:
            if isinstance(op, Primitive):
                ctx.out_state = apply_operator(
                    ctx.out_state, op.dagger(values), op.qubit_support
                )
                if isinstance(op, Parametric):
                    if values[op.param_name].requires_grad:
                        mu = apply_operator(
                            ctx.out_state, op.jacobian(values), op.qubit_support
                        )
                        grad = grad_out * 2 * inner_prod(ctx.projected_state, mu).real
                    if grads_dict[op.param_name] is not None:
                        grads_dict[op.param_name] += grad
                    else:
                        grads_dict[op.param_name] = grad

                ctx.projected_state = apply_operator(
                    ctx.projected_state, op.dagger(values), op.qubit_support
                )
            else:
                raise NotImplementedError(
                    f"AdjointExpectation does not support operation: {type(op)}."
                )
        return (None, None, None, None, *grads_dict.values())


def expectation(
    circuit: QuantumCircuit,
    state: Tensor,
    values: dict[str, Tensor],
    observable: Hamiltonian,
    diff_mode: DiffMode = DiffMode.AD,
) -> Tensor:
    """Compute the expectation value of the circuit given a state and observable.
    Arguments:
        circuit: QuantumCircuit instance
        state: An input state
        values: A dictionary of parameter values
        observable: Hamiltonian representing the observable
        diff_mode: The differentiation mode
    Returns:
        A expectation value.
    """
    if observable is None:
        raise ValueError("Please provide an observable to compute expectation.")
    if state is None:
        state = circuit.init_state(batch_size=1)
    if diff_mode == DiffMode.AD:
        state = circuit.run(state, values)
        return inner_prod(state, observable.run(state, values)).real
    elif diff_mode == DiffMode.ADJOINT:
        from pyqtorch.adjoint import AdjointExpectation

        return AdjointExpectation.apply(
            circuit, observable, state, values.keys(), *values.values()
        )
    else:
        raise ValueError(f"Requested diff_mode '{diff_mode}' not supported.")
