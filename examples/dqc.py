from __future__ import annotations

from functools import reduce
from itertools import product
from operator import add

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, exp, linspace, nn, ones_like, optim, rand, sin, tensor
from torch.autograd import grad

import pyqtorch as pyq
from pyqtorch.parametric import Parametric
from pyqtorch.utils import DiffMode

DIFF_MODE = DiffMode.AD
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PLOT = False
LEARNING_RATE = 0.01
N_QUBITS = 4
DEPTH = 3
VARIABLES = ("x", "y")
N_POINTS = 150
N_EPOCHS = 1000


def hea(n_qubits: int, n_layers: int, param_name: str) -> list:
    ops = []
    for layer in range(n_layers):
        ops += [pyq.RX(i, f"{param_name}_0_{layer}_{i}") for i in range(n_qubits)]
        ops += [pyq.RY(i, f"{param_name}_1_{layer}_{i}") for i in range(n_qubits)]
        ops += [pyq.RX(i, f"{param_name}_2_{layer}_{i}") for i in range(n_qubits)]
        ops += [pyq.CNOT(i % n_qubits, (i + 1) % n_qubits) for i in range(n_qubits)]
    return ops


class Observable(pyq.QuantumCircuit):
    def __init__(self, n_qubits: int, operations: list[torch.nn.Module]):
        super().__init__(n_qubits, operations)

    def forward(self, state, values) -> torch.Tensor:
        return reduce(add, [op(state, values) for op in self.operations])


feature_map = [pyq.RX(i, "x") for i in range(N_QUBITS // 2)] + [
    pyq.RX(i, "y") for i in range(N_QUBITS // 2, N_QUBITS)
]
ansatz = hea(N_QUBITS, DEPTH, "theta")
observable = Observable(N_QUBITS, [pyq.Z(i) for i in range(N_QUBITS)])
param_dict = torch.nn.ParameterDict(
    {
        op.param_name: torch.rand(1, requires_grad=True)
        for op in ansatz
        if isinstance(op, Parametric)
    }
)
circ = pyq.QuantumCircuit(N_QUBITS, feature_map + ansatz)
circ = circ.to(DEVICE)
observable = observable.to(DEVICE)
param_dict = param_dict.to(DEVICE)
state = circ.init_state()


def exp_fn(inputs: torch.Tensor) -> torch.Tensor:
    return pyq.expectation(
        circ, state, {**param_dict, **{"x": inputs[:, 0], "y": inputs[:, 1]}}, observable, DIFF_MODE
    )


single_domain_torch = linspace(0, 1, steps=N_POINTS)
domain_torch = tensor(list(product(single_domain_torch, single_domain_torch)))


def calc_derivative(outputs, inputs) -> Tensor:
    """
    Returns the derivative of a function output.

    with respect to its inputs
    """
    if not inputs.requires_grad:
        inputs.requires_grad = True
    return grad(
        inputs=inputs,
        outputs=outputs,
        grad_outputs=ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


class DomainSampling(nn.Module):
    """
    Collocation points sampling from domains uses uniform random sampling.

    Problem-specific MSE loss function for solving the 2D Laplace equation.
    """

    def __init__(
        self,
        n_inputs: int = len(VARIABLES),
        n_colpoints: int = N_POINTS,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.n_colpoints = n_colpoints
        self.n_inputs = n_inputs
        self.device = device

    def sample(self) -> Tensor:
        return rand(size=(self.n_colpoints, self.n_inputs), device=self.device)

    def left_boundary(self) -> Tensor:  # u(0,y)=0
        sample = self.sample()
        sample[:, 0] = 0.0
        return exp_fn(sample).pow(2).mean()

    def right_boundary(self) -> Tensor:  # u(L,y)=0
        sample = self.sample()
        sample[:, 0] = 1.0
        return exp_fn(sample).pow(2).mean()

    def top_boundary(self) -> Tensor:  # u(x,H)=0
        sample = self.sample()
        sample[:, 1] = 1.0
        return exp_fn(sample).pow(2).mean()

    def bottom_boundary(self) -> Tensor:  # u(x,0)=f(x)
        sample = self.sample()
        sample[:, 1] = 0.0
        return (exp_fn(sample) - sin(np.pi * sample[:, 0])).pow(2).mean()

    def interior(self) -> Tensor:  # uxx+uyy=0
        sample = self.sample().requires_grad_()
        first_both = calc_derivative(exp_fn(sample), sample)
        second_both = calc_derivative(first_both, sample)
        return (second_both[:, 0] + second_both[:, 1]).pow(2).mean()


def torch_solve() -> np.ndarray:
    opt = optim.Adam(param_dict.values(), lr=LEARNING_RATE)
    sol = DomainSampling(n_inputs=len(VARIABLES), n_colpoints=N_POINTS, device=DEVICE)
    for _ in range(N_EPOCHS):
        opt.zero_grad()
        loss = (
            sol.left_boundary()
            + sol.right_boundary()
            + sol.top_boundary()
            + sol.bottom_boundary()
            + sol.interior()
        )
        loss.backward()
        opt.step()
    return exp_fn(domain_torch.to(DEVICE)).reshape(N_POINTS, N_POINTS).detach().cpu().numpy()


if __name__ == "__main__":
    dqc_sol_torch = torch_solve()
    analytic_sol = (
        (exp(-np.pi * domain_torch[:, 0]) * sin(np.pi * domain_torch[:, 1]))
        .reshape(N_POINTS, N_POINTS)
        .T
    ).numpy()
    if PLOT:
        fig, ax = plt.subplots(1, 2, figsize=(7, 7))
        ax[0].imshow(analytic_sol, cmap="turbo")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")
        ax[0].set_title("Analytical solution u(x,y)")
        ax[1].imshow(dqc_sol_torch, cmap="turbo")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")
        ax[1].set_title("Torch DQC")
        plt.show()
