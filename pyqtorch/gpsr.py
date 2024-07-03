from __future__ import annotations

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Tuple

import torch
from torch import Tensor, no_grad
from torch.autograd import Function

from pyqtorch.analog import Observable
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.utils import inner_prod, param_dict

PI = torch.pi


def general_psr(
    spectrum: Tensor, n_eqs: int | None = None, shift_prefac: float = 0.5
) -> Callable:
    diffs = spectrum - spectrum.reshape(-1, 1)
    sorted_unique_spectral_gaps = torch.unique(torch.abs(torch.tril(diffs)))

    # We have to filter out zeros
    sorted_unique_spectral_gaps = sorted_unique_spectral_gaps[
        sorted_unique_spectral_gaps > 0
    ]
    n_eqs = len(sorted_unique_spectral_gaps)
    sorted_unique_spectral_gaps = torch.tensor(list(sorted_unique_spectral_gaps))

    if n_eqs == 1:
        return single_gap_psr
    else:
        return partial(
            multi_gap_psr,
            spectral_gaps=sorted_unique_spectral_gaps,
            shift_prefac=shift_prefac,
        )


def single_gap_psr(
    expectation_fn: Callable[[dict[str, Tensor]], Tensor],
    param_dict: dict[str, Tensor],
    param_name: str,
    spectral_gap: Tensor = torch.tensor([2.0], dtype=torch.get_default_dtype()),
    shift: Tensor = torch.tensor([PI / 2.0], dtype=torch.get_default_dtype()),
) -> Tensor:
    """Implements single qubit Parameter Shift rule.

    Args:
        expectation_fn: backend-dependent function
        to calculate expectation value

        param_dict: dict storing parameters of parameterized blocks
        param_name: name of parameter with respect to that differentiation is performed

    Returns:
        Tensor: tensor containing derivative values
    """
    device = torch.device("cpu")
    try:
        device = [v.device for v in param_dict.values()][0]
    except Exception:
        pass
    spectral_gap = spectral_gap.to(device=device)
    shift = shift.to(device=device)
    # + pi/2 shift
    shifted_params = param_dict.copy()
    shifted_params[param_name] = shifted_params[param_name] + shift
    f_plus = expectation_fn(shifted_params)

    # - pi/2 shift
    shifted_params = param_dict.copy()
    shifted_params[param_name] = shifted_params[param_name] - shift
    f_min = expectation_fn(shifted_params)

    return spectral_gap * (f_plus - f_min) / (4 * torch.sin(spectral_gap * shift / 2))


def multi_gap_psr(
    expectation_fn: Callable[[dict[str, Tensor]], Tensor],
    param_dict: dict[str, Tensor],
    param_name: str,
    spectral_gaps: Tensor,
    shift_prefac: float = 0.5,
) -> Tensor:
    """Implements multi-gap multi-qubit Generalized Parameter Shift rule (GPSR).

    Args:
        expectation_fn (Callable[[dict[str, Tensor]], Tensor]): backend-dependent function
        to calculate expectation value

        param_dict (dict[str, Tensor]): dict storing parameters values of parameterized blocks
        param_name (str): name of parameter with respect to that differentiation is performed
        spectral_gaps (Tensor): tensor containing spectral gap values
        shift_prefac (float): prefactor governing the magnitude of parameter shift values -
        select smaller value if spectral gaps are large

    Returns:
        Tensor: tensor containing derivative values
    """
    n_eqs = len(spectral_gaps)
    batch_size = max(t.size(0) for t in param_dict.values())

    # get shift values
    shifts = shift_prefac * torch.linspace(PI / 2 - PI / 5, PI / 2 + PI / 5, n_eqs)
    device = torch.device("cpu")
    try:
        device = [v.device for v in param_dict.values()][0]
    except Exception:
        pass
    spectral_gaps = spectral_gaps.to(device=device)
    shifts = shifts.to(device=device)
    # calculate F vector and M matrix
    # (see: https://arxiv.org/pdf/2108.01218.pdf on p. 4 for definitions)
    F = []
    M = torch.empty((n_eqs, n_eqs)).to(device=device)
    n_obs = 1
    for i in range(n_eqs):
        # + shift
        shifted_params = param_dict.copy()
        shifted_params[param_name] = shifted_params[param_name] + shifts[i]
        f_plus = expectation_fn(shifted_params)

        # - shift
        shifted_params = param_dict.copy()
        shifted_params[param_name] = shifted_params[param_name] - shifts[i]
        f_minus = expectation_fn(shifted_params)

        F.append((f_plus - f_minus))

        # calculate M matrix
        for j in range(n_eqs):
            M[i, j] = 4 * torch.sin(shifts[i] * spectral_gaps[j] / 2)

    # get number of observables from expectation value tensor
    if f_plus.numel() > 1:
        n_obs = F[0].shape[1]

    # reshape F vector
    F = torch.stack(F).reshape(n_eqs, -1)

    # calculate R vector
    R = torch.linalg.solve(M, F)

    # calculate df/dx
    dfdx = torch.sum(spectral_gaps[:, None] * R, dim=0).reshape(batch_size, n_obs)

    return dfdx


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
        ctx.in_state = state
        values = param_dict(param_names, param_values)
        out_state = circuit.run(state, values)
        projected_state = observable.run(out_state, values)
        ctx.save_for_backward(*param_values)
        return inner_prod(out_state, projected_state).real

    @staticmethod
    @no_grad()
    def backward(ctx: Any, grad_out: Tensor) -> Tuple[None, ...]:
        param_values = ctx.saved_tensors
        params_map = param_dict(ctx.param_names, param_values)
        grads_dict = {k: None for k in params_map.keys()}

        return (None, None, None, None, *grads_dict.values())

    @staticmethod
    def construct_rules(
        self,
        circuit: QuantumCircuit,
        observable: Observable,
    ) -> dict[str, Callable]:
        param_to_psr: OrderedDict = OrderedDict()
        return param_to_psr
