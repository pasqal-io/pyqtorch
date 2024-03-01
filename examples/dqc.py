from __future__ import annotations

from functools import reduce
from itertools import product
from operator import add
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, exp, linspace, ones_like, optim, rand, sin, tensor
from torch.autograd import grad

import pyqtorch as pyq
from pyqtorch.parametric import Parametric
from pyqtorch.utils import DiffMode

DIFF_MODE = DiffMode.ADJOINT
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PLOT = True
LEARNING_RATE = 0.01
N_QUBITS = 4
DEPTH = 3
VARIABLES = ("x", "y")
X_POS = 0
Y_POS = 1
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


class TotalMagnetization(pyq.QuantumCircuit):
    def __init__(self, n_qubits: int):
        super().__init__(n_qubits, [pyq.Z(i) for i in range(n_qubits)])

    def forward(self, state, values) -> torch.Tensor:
        return reduce(add, [op(state, values) for op in self.operations])


class DomainSampling(torch.nn.Module):
    def __init__(
        self, exp_fn: Callable[[Tensor], Tensor], n_inputs: int, n_points: int, device: torch.device
    ) -> None:
        super().__init__()
        self.exp_fn = exp_fn
        self.n_inputs = n_inputs
        self.n_points = n_points
        self.device = device

    def sample(self, requires_grad: bool = False) -> Tensor:
        return rand((self.n_points, self.n_inputs), requires_grad=requires_grad, device=self.device)

    def left_boundary(self) -> Tensor:  # u(0,y)=0
        sample = self.sample()
        sample[:, X_POS] = 0.0
        return self.exp_fn(sample).pow(2).mean()

    def right_boundary(self) -> Tensor:  # u(L,y)=0
        sample = self.sample()
        sample[:, X_POS] = 1.0
        return self.exp_fn(sample).pow(2).mean()

    def top_boundary(self) -> Tensor:  # u(x,H)=0
        sample = self.sample()
        sample[:, Y_POS] = 1.0
        return self.exp_fn(sample).pow(2).mean()

    def bottom_boundary(self) -> Tensor:  # u(x,0)=f(x)
        sample = self.sample()
        sample[:, Y_POS] = 0.0
        return (self.exp_fn(sample) - sin(np.pi * sample[:, 0])).pow(2).mean()

    def interior(self) -> Tensor:  # uxx+uyy=0
        sample = self.sample(requires_grad=True)
        f = self.exp_fn(sample)
        dfdxy = grad(
            f,
            sample,
            ones_like(f),
            create_graph=True,
            retain_graph=True,
        )[0]
        dfdxxdyy = grad(
            dfdxy,
            sample,
            ones_like(dfdxy),
            retain_graph=True,
        )[0]

        return (dfdxxdyy[:, X_POS] + dfdxxdyy[:, Y_POS]).pow(2).mean()


feature_map = [pyq.RX(i, VARIABLES[X_POS]) for i in range(N_QUBITS // 2)] + [
    pyq.RX(i, VARIABLES[Y_POS]) for i in range(N_QUBITS // 2, N_QUBITS)
]
ansatz = hea(N_QUBITS, DEPTH, "theta")
param_dict = torch.nn.ParameterDict(
    {
        op.param_name: torch.rand(1, requires_grad=True)
        for op in ansatz
        if isinstance(op, Parametric)
    }
)
circ = pyq.QuantumCircuit(N_QUBITS, feature_map + ansatz).to(DEVICE)
observable = TotalMagnetization(N_QUBITS).to(DEVICE)
param_dict = param_dict.to(DEVICE)
state = circ.init_state()


def exp_fn(inputs: torch.Tensor) -> torch.Tensor:
    return pyq.expectation(
        circ,
        state,
        {**param_dict, **{VARIABLES[X_POS]: inputs[:, 0], VARIABLES[Y_POS]: inputs[:, 1]}},
        observable,
        DIFF_MODE,
    )


single_domain_torch = linspace(0, 1, steps=N_POINTS)
domain_torch = tensor(list(product(single_domain_torch, single_domain_torch)))

opt = optim.Adam(param_dict.values(), lr=LEARNING_RATE)
sol = DomainSampling(exp_fn, len(VARIABLES), N_POINTS, DEVICE)

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

dqc_sol = exp_fn(domain_torch.to(DEVICE)).reshape(N_POINTS, N_POINTS).detach().cpu().numpy()
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
    ax[1].imshow(dqc_sol, cmap="turbo")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].set_title("Torch DQC")
    plt.show()
