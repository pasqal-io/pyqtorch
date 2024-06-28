from __future__ import annotations

from logging import getLogger
from typing import Any, Tuple

from torch import Tensor, no_grad
from torch.autograd import Function

from pyqtorch.analog import Observable, Scale
from pyqtorch.apply import apply_operator
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.parametric import Parametric
from pyqtorch.primitive import Primitive
from pyqtorch.utils import DiffMode, inner_prod, param_dict

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
        observable: Observable,
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
            elif isinstance(op, Scale):
                if not len(op.operations) == 1 and isinstance(
                    op.operations[0], Primitive
                ):
                    logger.error(
                        "Adjoint can only be used on Scale with Primitive blocks."
                    )
                ctx.out_state = apply_operator(
                    ctx.out_state, op.dagger(values), op.qubit_support
                )
                scaled_pyq_op = op.operations[0]
                if (
                    isinstance(scaled_pyq_op, Parametric)
                    and values[scaled_pyq_op.param_name].requires_grad
                ):
                    mu = apply_operator(
                        ctx.out_state,
                        scaled_pyq_op.jacobian(values),
                        scaled_pyq_op.qubit_support,
                    )
                    grads_dict[scaled_pyq_op.param_name] = (
                        grad_out * 2 * inner_prod(ctx.projected_state, mu).real
                    )

                if values[op.param_name].requires_grad:
                    grads_dict[op.param_name] = grad_out * 2 * -values[op.param_name]
                ctx.projected_state = apply_operator(
                    ctx.projected_state, op.dagger(values), op.qubit_support
                )
            else:
                logger.error(
                    f"AdjointExpectation does not support operation: {type(op)}."
                )
        return (None, None, None, None, *grads_dict.values())


class GPSR(Function):
    r"""
    Implementation of the generalized parameter shift rule (Kyriienko et al.),
    which only works for quantum operations whose generator has a single gap
    in its eigenvalue spectrum, was generalized to work with arbitrary
    generators of quantum operations.

    For this, we define the differentiable function as quantum expectation value

    $$
    f(x) = \left\langle 0\right|\hat{U}^{\dagger}(x)\hat{C}\hat{U}(x)\left|0\right\rangle
    $$

    where $\hat{U}(x)={\rm exp}{\left( -i\frac{x}{2}\hat{G}\right)}$
    is the quantum evolution operator with generator $\hat{G}$ representing the structure
    of the underlying quantum circuit and $\hat{C}$ is the cost operator.
    Then using the eigenvalue spectrum $\left\{ \lambda_n\right\}$ of the generator $\hat{G}$
    we calculate the full set of corresponding unique non-zero spectral gaps
    $\left\{ \Delta_s\right\}$ (differences between eigenvalues).
    It can be shown that the final expression of derivative of $f(x)$
    is then given by the following expression:

    $\begin{equation}
    \frac{{\rm d}f\left(x\right)}{{\rm d}x}=\overset{S}{\underset{s=1}{\sum}}\Delta_{s}R_{s},
    \end{equation}$

    where $S$ is the number of unique non-zero spectral gaps and $R_s$ are real quantities that
    are solutions of a system of linear equations

    $\begin{equation}
    \begin{cases}
    F_{1} & =4\overset{S}{\underset{s=1}{\sum}}{\rm sin}
    \left(\frac{\delta_{1}\Delta_{s}}{2}\right)R_{s},\\
    F_{2} & =4\overset{S}{\underset{s=1}{\sum}}{\rm sin}
    \left(\frac{\delta_{2}\Delta_{s}}{2}\right)R_{s},\\
    & ...\\
    F_{S} & =4\overset{S}{\underset{s=1}{\sum}}{\rm sin}
    \left(\frac{\delta_{M}\Delta_{s}}{2}\right)R_{s}.
    \end{cases}
    \end{equation}$

    Here $F_s=f(x+\delta_s)-f(x-\delta_s)$ denotes the difference between values
    of functions evaluated at shifted arguments $x\pm\delta_s$.

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
        values = param_dict(param_names, param_values)
        ctx.out_state = circuit.run(state, values)
        ctx.projected_state = observable.run(ctx.out_state, values)
        ctx.save_for_backward(*param_values)
        return inner_prod(ctx.out_state, ctx.projected_state).real

    @staticmethod
    @no_grad()
    def backward(ctx: Any, grad_out: Tensor) -> Tuple[None, ...]:
        # param_values = ctx.saved_tensors
        # params_map = param_dict(ctx.param_names, param_values)
        # grads_dict = {k: None for k in values.keys()}

        # def expectation_fn(params: dict[str, Tensor]) -> Tensor:
        #     return GPSR.apply(
        #         ctx.expectation_fn,
        #         ctx.param_psrs,
        #         params_map.keys(),
        #         *params_map.values(),
        #     )

        # def vjp(psr: Callable, name: str) -> Tensor:
        #     """Sums over gradients corresponding to different observables.
        #     """
        #     return (grad_out * psr(expectation_fn, params_map, name)).sum(dim=1)

        raise NotImplementedError


def expectation(
    circuit: QuantumCircuit,
    state: Tensor,
    values: dict[str, Tensor],
    observable: Observable,
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
        logger.error("Please provide an observable to compute expectation.")
    if state is None:
        state = circuit.init_state(batch_size=1)
    if diff_mode == DiffMode.AD:
        state = circuit.run(state, values)
        return inner_prod(state, observable.run(state, values)).real
    elif diff_mode == DiffMode.ADJOINT:
        return AdjointExpectation.apply(
            circuit, observable, state, values.keys(), *values.values()
        )
    elif diff_mode == DiffMode.GPSR:
        return GPSR.apply(circuit, observable, state, values.keys(), *values.values())
    else:
        logger.error(f"Requested diff_mode '{diff_mode}' not supported.")
