from __future__ import annotations

from logging import getLogger
from typing import Any, Tuple

import torch
from torch import Tensor, no_grad
from torch.autograd import Function

import pyqtorch as pyq
from pyqtorch.analog import HamiltonianEvolution, Observable, Scale
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.parametric import Parametric
from pyqtorch.utils import inner_prod, param_dict

logger = getLogger(__name__)


class PSRExpectation(Function):
    r"""
    Implementation of the generalized parameter shift rule.

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
        ctx.projected_state = observable(ctx.out_state, values)
        ctx.save_for_backward(*param_values)
        return inner_prod(ctx.out_state, ctx.projected_state).real

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> Tuple[None, ...]:
        check_support_psr(ctx.circuit)
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
                        # reset values
                        copied_values[op.param_name] += shift

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


def check_support_psr(circuit: QuantumCircuit):
    """Checking that circuit has only compatible operations for PSR.

    Args:
        circuit (QuantumCircuit): Circuit to check.

    Raises:
        ValueError: When circuit contains Scale or HamiltonianEvolution.
    """
    for op in circuit.operations:
        if isinstance(op, Scale) or isinstance(op, HamiltonianEvolution):
            raise ValueError(
                f"PSR is not applicable as circuit contains an operation of type: {type(op)}."
            )
