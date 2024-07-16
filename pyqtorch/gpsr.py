from __future__ import annotations

from logging import getLogger
from typing import Any, Tuple

import torch
from torch import Tensor, no_grad
from torch.autograd import Function

from pyqtorch.analog import HamiltonianEvolution, Observable, Scale
from pyqtorch.circuit import QuantumCircuit, Sequence
from pyqtorch.embed import Embedding
from pyqtorch.parametric import Parametric
from pyqtorch.utils import inner_prod, param_dict

logger = getLogger(__name__)


class PSRExpectation(Function):
    r"""
    Implementation of the generalized parameter shift rule.

    Note that only operations with two distinct eigenvalues
    from their generator (i.e., compatible with single_gap_shift)
    are supported at the moment.

    Compared to the original parameter shift rule
    which only works for quantum operations whose generator has a single gap
    in its eigenvalue spectrum, GPSR works with arbitrary
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
        state: Tensor,
        observable: Observable,
        embedding: Embedding,
        param_names: list[str],
        *param_values: Tensor,
    ) -> Tensor:
        if embedding is not None:
            logger.error("GPSR does not support Embedding.")
        ctx.circuit = circuit
        ctx.observable = observable
        ctx.param_names = param_names
        ctx.state = state
        ctx.embedding = embedding
        values = param_dict(param_names, param_values)
        ctx.out_state = circuit.run(state, values)
        ctx.projected_state = observable.run(ctx.out_state, values)
        ctx.save_for_backward(*param_values)
        return inner_prod(ctx.out_state, ctx.projected_state).real

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> Tuple[None, ...]:
        """The PSRExpectation backward call.

        Note that only operations with two distinct eigenvalues
        from their generator (i.e., compatible with single_gap_shift)
        are supported at the moment.

        Arguments:
            ctx (Any): Context object for accessing stored information.
            grad_out (Tensor): Current jacobian tensor.

        Returns:
            A tuple of updated jacobian tensor.

        Raises:
            ValueError: When operation is not supported.
        """

        values = param_dict(ctx.param_names, ctx.saved_tensors)
        shift = torch.tensor(torch.pi) / 2.0

        def expectation_fn(values: dict[str, Tensor]) -> Tensor:
            """Use the PSRExpectation for nested grad calls.

            Arguments:
                values: Dictionary with parameter values.

            Returns:
                Expectation evaluation.
            """
            return PSRExpectation.apply(
                ctx.circuit,
                ctx.state,
                ctx.observable,
                ctx.embedding,
                values.keys(),
                *values.values(),
            )

        def single_gap_shift(
            param_name: str,
            values: dict[str, torch.Tensor],
            spectral_gap: torch.Tensor,
            shift: torch.Tensor = torch.tensor(torch.pi) / 2.0,
        ) -> torch.Tensor:
            """Implements single gap PSR rule.

            Args:
                param_name: Name of the parameter to apply PSR.
                values: Dictionary with parameter values.
                spectral_gap: Spectral gap value for PSR.
                shift: Shift value. Defaults to torch.tensor(torch.pi)/2.0.

            Returns:
                Gradient evaluation for param_name.
            """
            shifted_values = values.copy()
            shifted_values[param_name] = shifted_values[param_name] + shift
            f_plus = expectation_fn(shifted_values)
            shifted_values[param_name] = shifted_values[param_name] - 2 * shift
            f_minus = expectation_fn(shifted_values)
            return (
                spectral_gap
                * (f_plus - f_minus)
                / (4 * torch.sin(spectral_gap * shift / 2))
            )

        def multi_gap_shift(*args, **kwargs) -> Tensor:
            """Implements multi gap PSR rule."""
            raise NotImplementedError("Multi-gap is not yet supported.")

        def vjp(operation: Parametric, values: dict[str, Tensor]) -> Tensor:
            """Vector-jacobian product between `grad_out` and jacobians of parameters.

            Args:
                operation: Parametric operation to compute PSR.
                values: Dictionary with parameter values.

            Returns:
                Updated jacobian by PSR.
            """
            psr_fn = (
                multi_gap_shift if len(operation.spectral_gap) > 1 else single_gap_shift
            )

            return grad_out * psr_fn(  # type: ignore[operator]
                operation.param_name, values, operation.spectral_gap, shift
            )

        grads = {p: None for p in ctx.param_names}
        for op in ctx.circuit.flatten():
            if isinstance(op, Parametric) and values[op.param_name].requires_grad:  # type: ignore[index]
                if grads[op.param_name] is not None:
                    grads[op.param_name] += vjp(op, values)
                else:
                    grads[op.param_name] = vjp(op, values)

        return (None, None, None, None, None, *[grads[p] for p in ctx.param_names])


def check_support_psr(circuit: QuantumCircuit):
    """Checking that circuit has only compatible operations for PSR.

    Args:
        circuit (QuantumCircuit): Circuit to check.

    Raises:
        ValueError: When circuit contains Scale, HamiltonianEvolution,
                    or one operation has more than two eigenvalues (multi-gap),
                    or a param_name is used multiple times in the circuit.
    """

    param_names = list()
    for op in circuit.operations:
        if isinstance(op, Scale) or isinstance(op, HamiltonianEvolution):
            raise ValueError(
                f"PSR is not applicable as circuit contains an operation of type: {type(op)}."
            )
        if isinstance(op, Sequence):
            for subop in op.flatten():
                if isinstance(subop, Parametric):
                    if isinstance(subop.param_name, str):
                        if len(subop.spectral_gap) > 1:
                            raise NotImplementedError("Multi-gap is not yet supported.")
                        param_names.append(subop.param_name)

        elif isinstance(op, Parametric):
            if len(op.spectral_gap) > 1:
                raise NotImplementedError("Multi-gap is not yet supported.")
            if isinstance(op.param_name, str):
                param_names.append(op.param_name)
        else:
            continue

    if len(param_names) > len(set(param_names)):
        raise ValueError(
            "PSR is not supported when using a same param_name in different operations."
        )
