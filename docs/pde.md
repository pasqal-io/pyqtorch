`pyqtorch` can also be used to implement [DQC](https://arxiv.org/abs/2011.10395) to solve a partial differential equation.

```python exec="on" source="material-block" html="1"
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
from pyqtorch.composite import hea
from pyqtorch import CNOT, RX, RY, QuantumCircuit, Z, expectation, Sequence, Merge, Add, Observable
from pyqtorch.primitives import Parametric
from pyqtorch.utils import DiffMode

DIFF_MODE = DiffMode.AD
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# We can also choose the precision we want to train on
COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32
LR = .15
N_QUBITS = 4
DEPTH = 3
VARIABLES = ("x", "y")
N_VARIABLES = len(VARIABLES)
X_POS, Y_POS = [i for i in range(N_VARIABLES)]
BATCH_SIZE = 250
N_EPOCHS = 750


class DomainSampling(torch.nn.Module):
    def __init__(
        self, exp_fn: Callable[[Tensor], Tensor], n_inputs: int, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.exp_fn = exp_fn
        self.n_inputs = n_inputs
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

    def sample(self, requires_grad: bool = False) -> Tensor:
        return rand((self.batch_size, self.n_inputs), requires_grad=requires_grad, device=self.device, dtype=self.dtype)

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
        )[0]
        dfdxxdyy = grad(
            dfdxy,
            sample,
            ones_like(dfdxy),
        )[0]

        return (dfdxxdyy[:, X_POS] + dfdxxdyy[:, Y_POS]).pow(2).mean()


feature_map = [RX(i, VARIABLES[X_POS]) for i in range(N_QUBITS // 2)] + [
    RX(i, VARIABLES[Y_POS]) for i in range(N_QUBITS // 2, N_QUBITS)
]
ansatz, params = hea(N_QUBITS, DEPTH, "theta")
circ = QuantumCircuit(N_QUBITS, feature_map + ansatz).to(device=DEVICE, dtype=COMPLEX_DTYPE)
sumZ_obs = Observable([Z(i) for i in range(N_QUBITS)]).to(device=DEVICE, dtype=COMPLEX_DTYPE)
params = params.to(device=DEVICE, dtype=REAL_DTYPE)
state = circ.init_state()


def exp_fn(inputs: Tensor) -> Tensor:
    return expectation(
        circ,
        state,
        {**params, **{VARIABLES[X_POS]: inputs[:, X_POS], VARIABLES[Y_POS]: inputs[:, Y_POS]}},
        sumZ_obs,
        DIFF_MODE,
    )


single_domain_torch = linspace(0, 1, steps=BATCH_SIZE)
domain_torch = tensor(list(product(single_domain_torch, single_domain_torch)))

opt = optim.Adam(params.values(), lr=LR)
sol = DomainSampling(exp_fn, len(VARIABLES), BATCH_SIZE, DEVICE, REAL_DTYPE)

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

dqc_sol = exp_fn(domain_torch.to(DEVICE)).reshape(BATCH_SIZE, BATCH_SIZE).detach().cpu().numpy()
analytic_sol = (
    (exp(-np.pi * domain_torch[:, X_POS]) * sin(np.pi * domain_torch[:, Y_POS]))
    .reshape(BATCH_SIZE, BATCH_SIZE)
    .T
).numpy()


fig, ax = plt.subplots(1, 2, figsize=(7, 7))
ax[0].imshow(analytic_sol, cmap="turbo")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title("Analytical solution u(x,y)")
ax[1].imshow(dqc_sol, cmap="turbo")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_title("DQC solution")
from io import StringIO  # markdown-exec: hide
from matplotlib.figure import Figure  # markdown-exec: hide
def fig_to_html(fig: Figure) -> str:  # markdown-exec: hide
    buffer = StringIO()  # markdown-exec: hide
    fig.savefig(buffer, format="svg")  # markdown-exec: hide
    return buffer.getvalue()  # markdown-exec: hide
print(fig_to_html(plt.gcf())) # markdown-exec: hide
```
