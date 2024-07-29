from __future__ import annotations

from logging import getLogger
from typing import Any, Callable, Tuple

import torch
from torch import Tensor, no_grad
from torch.autograd import Function

from pyqtorch.analog import HamiltonianEvolution, Observable, Scale
from pyqtorch.circuit import QuantumCircuit, Sequence
from pyqtorch.embed import Embedding
from pyqtorch.matrices import DEFAULT_REAL_DTYPE
from pyqtorch.parametric import Parametric
from pyqtorch.utils import param_dict

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
        state: A state in the form of [2 * n_qubits + [batch_size]]
        observable: An hermitian operator.
        embedding: An optional instance of `Embedding`.
        expectation_method: Callable for computing expectations.
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
        expectation_method: Callable,
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
        ctx.save_for_backward(*param_values)
        ctx.expectation_method = expectation_method
        return expectation_method(circuit, state, observable, values, embedding)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> Tuple[None, ...]:
        """The PSRExpectation backward call.

        Arguments:
            ctx (Any): Context object for accessing stored information.
            grad_out (Tensor): Current jacobian tensor.

        Returns:
            A tuple of updated jacobian tensor.

        Raises:
            ValueError: When operation is not supported.
        """

        values = param_dict(ctx.param_names, ctx.saved_tensors)
        shift_pi2 = torch.tensor(torch.pi) / 2.0
        shift_multi = 0.5

        dtype_values = DEFAULT_REAL_DTYPE
        device = torch.device("cpu")
        try:
            dtype_values, device = [(v.dtype, v.device) for v in values.values()][0]
        except Exception:
            pass

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
                ctx.expectation_method,
                values.keys(),
                *values.values(),
            )

        def single_gap_shift(
            param_name: str,
            values: dict[str, Tensor],
            spectral_gap: Tensor,
            shift: Tensor = torch.tensor(torch.pi) / 2.0,
        ) -> Tensor:
            """Implements single gap PSR rule.

            Args:
                param_name: Name of the parameter to apply PSR.
                values: Dictionary with parameter values.
                spectral_gap: Spectral gap value for PSR.
                shift: Shift value. Defaults to torch.tensor(torch.pi)/2.0.

            Returns:
                Gradient evaluation for param_name.
            """

            # device conversions
            spectral_gap = spectral_gap.to(device=device)
            shift = shift.to(device=device)

            # apply shift rule
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

        def multi_gap_shift(
            param_name: str,
            values: dict[str, Tensor],
            spectral_gaps: Tensor,
            shift_prefac: float = 0.5,
        ) -> Tensor:
            """Implement multi gap PSR rule.

            See Kyriienko1 and Elfving, 2021 for details:
            https://arxiv.org/pdf/2108.01218.pdf

            Args:
                param_name: Name of the parameter to apply PSR.
                values: Dictionary with parameter values.
                spectral_gaps: Spectral gaps value for PSR.
                shift_prefac: Shift prefactor value for PSR shifts.
                Defaults to torch.tensor(0.5).

            Returns:
                Gradient evaluation for param_name.
            """
            n_eqs = len(spectral_gaps)
            dtype = torch.promote_types(dtype_values, spectral_gaps.dtype)
            spectral_gaps = spectral_gaps.to(device=device)
            PI = torch.tensor(torch.pi, dtype=dtype)
            shifts = shift_prefac * torch.linspace(
                PI / 2.0 - PI / 5.0, PI / 2.0 + PI / 5.0, n_eqs, dtype=dtype
            )
            shifts = shifts.to(device=device)

            # calculate F vector and M matrix
            # (see: https://arxiv.org/pdf/2108.01218.pdf on p. 4 for definitions)
            F = []
            M = torch.empty((n_eqs, n_eqs), dtype=dtype).to(device=device)
            batch_size = 1
            shifted_params = values.copy()
            for i in range(n_eqs):
                # + shift
                shifted_params[param_name] = shifted_params[param_name] + shifts[i]
                f_plus = expectation_fn(shifted_params)

                # - shift
                shifted_params[param_name] = shifted_params[param_name] - 2 * shifts[i]
                f_minus = expectation_fn(shifted_params)
                shifted_params[param_name] = shifted_params[param_name] + shifts[i]
                F.append((f_plus - f_minus))

                # calculate M matrix
                for j in range(n_eqs):
                    M[i, j] = 4 * torch.sin(shifts[i] * spectral_gaps[j] / 2)

            # get number of observables from expectation value tensor
            if f_plus.numel() > 1:
                batch_size = F[0].shape[0]

            F = torch.stack(F).reshape(n_eqs, -1)
            R = torch.linalg.solve(M, F)
            dfdx = torch.sum(spectral_gaps * R, dim=0).reshape(batch_size)
            return dfdx

        def vjp(operation: Parametric, values: dict[str, Tensor]) -> Tensor:
            """Vector-jacobian product between `grad_out` and jacobians of parameters.

            Args:
                operation: Parametric operation to compute PSR.
                values: Dictionary with parameter values.

            Returns:
                Updated jacobian by PSR.
            """
            psr_fn, shift = (
                (multi_gap_shift, shift_multi)
                if len(operation.spectral_gap) > 1
                else (single_gap_shift, shift_pi2)
            )
            return grad_out * psr_fn(  # type: ignore[operator]
                operation.param_name,  # type: ignore
                values,
                operation.spectral_gap,
                shift,
            )

        grads = {p: None for p in ctx.param_names}
        for op in ctx.circuit.flatten():
            if isinstance(op, Parametric) and isinstance(op.param_name, str):
                if values[op.param_name].requires_grad:
                    if grads[op.param_name] is not None:
                        grads[op.param_name] += vjp(op, values)
                    else:
                        grads[op.param_name] = vjp(op, values)

        return (
            None,
            None,
            None,
            None,
            None,
            None,
            *[grads[p] for p in ctx.param_names],
        )


def check_support_psr(circuit: QuantumCircuit):
    """Checking that circuit has only compatible operations for PSR.

    Args:
        circuit (QuantumCircuit): Circuit to check.

    Raises:
        ValueError: When circuit contains Scale, HamiltonianEvolution,
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
                if isinstance(subop, Scale) or isinstance(subop, HamiltonianEvolution):
                    raise ValueError(
                        f"PSR is not applicable as circuit contains \
                        an operation of type: {type(subop)}."
                    )
                if isinstance(subop, Parametric):
                    if isinstance(subop.param_name, str):
                        param_names.append(subop.param_name)
        elif isinstance(op, Parametric):
            if isinstance(op.param_name, str):
                param_names.append(op.param_name)
        else:
            continue

    if len(param_names) > len(set(param_names)):
        raise ValueError(
            "PSR is not supported when using a same param_name in different operations."
        )
