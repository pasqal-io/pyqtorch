# Welcome to pyqtorch

**pyqtorch** is a state vector simulator designed for quantum machine learning written in [PyTorch](https://pytorch.org/). It allows for building fully differentiable quantum circuits comprised of both digital and analog operations using a intuitive [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)-based API.

## Setup

To install `pyqtorch` , you can go into any virtual environment of your
choice and install it normally with `pip`:

```bash
pip install pyqtorch
```

## Digital Operations

`pyqtorch` implements a large selection of both primitive and parametric single to n-qubit, digital quantum gates.

Let's have a look at primitive gates first.

```python exec="on" source="material-block"
import torch
from pyqtorch import X, CNOT, random_state

x = X(0)
state = random_state(n_qubits=2)

new_state = x(state)

cnot = CNOT(0,1)
new_state= cnot(state)
```

Parametric gates can be initialized with or without a `param_name`. In the former case, a dictionary containing the `param_name` and a `torch.Tensor` for the parameter is expected when calling the forward method of the gate.

```python exec="on" source="material-block"
import torch
from pyqtorch import X, RX, CNOT, CRX, random_state

state = random_state(n_qubits=2)

rx_with_param = RX(0, 'theta')

theta = torch.rand(1)
values = {'theta': theta}
new_state = rx_with_param(state, values)

crx = CRX(0, 1, 'theta')
new_state = crx(state, values)
```

However, if you want to run a quick state vector simulation, you can initialize parametric gates without passing a `param_name`, in which case the forward method of the gate will simply expect a `torch.Tensor`.


```python exec="on" source="material-block"
import torch
from pyqtorch import RX, random_state

state = random_state(n_qubits=2)
rx = RX(0)
new_state = rx(state, torch.rand(1))
```

## Analog Operations

`pyqtorch` also contains a `analog` module which allows for global state evolution through the `HamiltonianEvolution` class. Note that it only accepts a `torch.Tensor` as a generator which is expected to be an Hermitian matrix. To build arbitrary Pauli hamiltonians, we recommend using [Qadence](https://pasqal-io.github.io/qadence/v1.0.3/tutorials/hamiltonians/).

```python exec="on" source="material-block" html="1"
import torch
from pyqtorch import uniform_state, HamiltonianEvolution, is_normalized
from pyqtorch.matrices import DEFAULT_MATRIX_DTYPE

n_qubits = 4

# Random hermitian hamiltonian
matrix = torch.rand(2**n_qubits, 2**n_qubits, dtype=DEFAULT_MATRIX_DTYPE)
hermitian_matrix = matrix + matrix.T.conj()

# To be evolved for a batch of times
t_list = torch.tensor([0.0, 0.5, 1.0, 2.0])

hamiltonian_evolution = HamiltonianEvolution(qubit_support=[i for i in range(n_qubits)], n_qubits=n_qubits)

# Starting from a uniform state
psi_start = uniform_state(n_qubits)

# Returns an evolved state at each time value
psi_end = hamiltonian_evolution(
    hamiltonian=hermitian_matrix,
    time_evolution=t_list,
    state = psi_start)

assert is_normalized(psi_end, atol=1e-05)
```

## Circuits

Using digital and analog operations, you can can build fully differentiable quantum circuits using the `QuantumCircuit` class; note that the default differentiation mode in pyqtorch is using torch.autograd.

```python exec="on" source="material-block"
import torch
import pyqtorch as pyq

rx = pyq.RX(0, param_name="theta")
y = pyq.Y(0)
cnot = pyq.CNOT(0, 1)
ops = [rx, y, cnot]
n_qubits = 2
circ = pyq.QuantumCircuit(n_qubits, ops)
state = pyq.random_state(n_qubits)
theta = torch.rand(1, requires_grad=True)
obs = pyq.QuantumCircuit(n_qubits, [pyq.Z(0)])
expval = pyq.expectation(circ, state, {"theta": theta}, obs)
dfdtheta = torch.autograd.grad(expval, theta, torch.ones_like(expval))
```

## Adjoint Differentiation

`pyqtorch` also offers a [adjoint differentiation mode](https://arxiv.org/abs/2009.02823) which can be used through the `expectation` method.

```python exec="on" source="material-block"
import pyqtorch as pyq
import torch
from pyqtorch.utils import DiffMode

n_qubits = 3
batch_size = 1

rx = pyq.RX(0, param_name="x")
cnot = pyq.CNOT(1, 2)
ops = [rx, cnot]
n_qubits = 3
circ = pyq.QuantumCircuit(n_qubits, ops)

obs = pyq.QuantumCircuit(n_qubits, [pyq.Z(0)])
state = pyq.zero_state(n_qubits)

values_ad = {"x": torch.tensor([torch.pi / 2], requires_grad=True)}
values_adjoint = {"x": torch.tensor([torch.pi / 2], requires_grad=True)}
exp_ad = pyq.expectation(circ, state, values_ad, obs, DiffMode.AD)
exp_adjoint = pyq.expectation(circ, state, values_adjoint, obs, DiffMode.ADJOINT)

dfdx_ad = torch.autograd.grad(exp_ad, tuple(values_ad.values()), torch.ones_like(exp_ad))

dfdx_adjoint = torch.autograd.grad(
    exp_adjoint, tuple(values_adjoint.values()), torch.ones_like(exp_adjoint)
)

assert len(dfdx_ad) == len(dfdx_adjoint)
for i in range(len(dfdx_ad)):
    assert torch.allclose(dfdx_ad[i], dfdx_adjoint[i])
```

## Fitting a nonlinear function

Let's have a look at how the `QuantumCircuit` can be used to fit a function.

```python exec="on" source="material-block" html="1"
from __future__ import annotations

from operator import add
from functools import reduce
import torch
import pyqtorch as pyq
from pyqtorch.utils import DiffMode
from pyqtorch.parametric import Parametric
import matplotlib.pyplot as plt

from torch.nn.functional import mse_loss

# We can train on GPU if available
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# We can also choose the precision we want to train on
COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32

# Target function and some training data
fn = lambda x, degree: .05 * reduce(add, (torch.cos(i*x) + torch.sin(i*x) for i in range(degree)), 0)
x = torch.linspace(0, 10, 100)
y = fn(x, 5)


def hea(n_qubits: int, n_layers: int, param_name: str) -> list:
    ops = []
    for l in range(n_layers):
        ops += [pyq.RX(i, f'{param_name}_0_{l}_{i}') for i in range(n_qubits)]
        ops += [pyq.RY(i, f'{param_name}_1_{l}_{i}') for i in range(n_qubits)]
        ops += [pyq.RX(i, f'{param_name}_2_{l}_{i}') for i in range(n_qubits)]
        ops += [pyq.CNOT(i % n_qubits, (i+1) % n_qubits) for i in range(n_qubits)]
    return ops

n_qubits = 5
n_layers = 3
diff_mode = DiffMode.ADJOINT
# Lets define a feature map to encode our 'x' values
feature_map = [pyq.RX(i, f'x') for i in range(n_qubits)]
# To fit the function, we define a hardware-efficient ansatz with tunable parameters
ansatz = hea(n_qubits, n_layers, 'theta')
observable = pyq.QuantumCircuit(n_qubits, [pyq.Z(0)])
param_dict = torch.nn.ParameterDict({op.param_name: torch.rand(1, requires_grad=True) for op in ansatz if isinstance(op, Parametric)})
circ = pyq.QuantumCircuit(n_qubits, feature_map + ansatz)
# Lets move all necessary components to the DEVICE
circ = circ.to(device=DEVICE, dtype=COMPLEX_DTYPE)
observable = observable.to(device=DEVICE, dtype=COMPLEX_DTYPE)
param_dict = param_dict.to(device=DEVICE, dtype=REAL_DTYPE)
x, y = x.to(device=DEVICE, dtype=REAL_DTYPE), y.to(device=DEVICE, dtype=REAL_DTYPE)
state = circ.init_state()

def exp_fn(param_dict: dict[str, torch.Tensor], inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return pyq.expectation(circ, state, {**param_dict,**inputs}, observable, diff_mode)

with torch.no_grad():
    y_init = exp_fn(param_dict, {'x': x})

# We need to set 'foreach' False since Adam doesnt support float64 on CUDA devices
optimizer = torch.optim.Adam(param_dict.values(), lr=.01, foreach=False)
epochs = 300

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = exp_fn(param_dict, {'x': x})
    loss = mse_loss(y, y_pred)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    y_final = exp_fn(param_dict, {'x': x})

plt.plot(x.numpy(), y.numpy(), label="truth")
plt.plot(x.numpy(), y_init.numpy(), label="initial")
plt.plot(x.numpy(), y_final.numpy(), "--", label="final", linewidth=3)
plt.legend()
from io import StringIO  # markdown-exec: hide
from matplotlib.figure import Figure  # markdown-exec: hide
def fig_to_html(fig: Figure) -> str:  # markdown-exec: hide
    buffer = StringIO()  # markdown-exec: hide
    fig.savefig(buffer, format="svg")  # markdown-exec: hide
    return buffer.getvalue()  # markdown-exec: hide
print(fig_to_html(plt.gcf())) # markdown-exec: hide
```

## Fitting a partial differential equation using DQC

Finally, we show how to implement [DQC](https://arxiv.org/abs/2011.10395) to solve a partial differential equation using `pyqtorch`.

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

from pyqtorch import CNOT, RX, RY, QuantumCircuit, Z, expectation
from pyqtorch.parametric import Parametric
from pyqtorch.utils import DiffMode

DIFF_MODE = DiffMode.AD
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# We can also choose the precision we want to train on
COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32
PLOT = False
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
        ops += [RX(i, f"{param_name}_0_{layer}_{i}") for i in range(n_qubits)]
        ops += [RY(i, f"{param_name}_1_{layer}_{i}") for i in range(n_qubits)]
        ops += [RX(i, f"{param_name}_2_{layer}_{i}") for i in range(n_qubits)]
        ops += [CNOT(i % n_qubits, (i + 1) % n_qubits) for i in range(n_qubits)]
    return ops


class TotalMagnetization(QuantumCircuit):
    def __init__(self, n_qubits: int):
        super().__init__(n_qubits, [Z(i) for i in range(n_qubits)])

    def forward(self, state, values) -> Tensor:
        return reduce(add, [op(state, values) for op in self.operations])


class DomainSampling(torch.nn.Module):
    def __init__(
        self, exp_fn: Callable[[Tensor], Tensor], n_inputs: int, n_points: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.exp_fn = exp_fn
        self.n_inputs = n_inputs
        self.n_points = n_points
        self.device = device
        self.dtype = dtype

    def sample(self, requires_grad: bool = False) -> Tensor:
        return rand((self.n_points, self.n_inputs), requires_grad=requires_grad, device=self.device, dtype=self.dtype)

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


feature_map = [RX(i, VARIABLES[X_POS]) for i in range(N_QUBITS // 2)] + [
    RX(i, VARIABLES[Y_POS]) for i in range(N_QUBITS // 2, N_QUBITS)
]
ansatz = hea(N_QUBITS, DEPTH, "theta")
param_dict = torch.nn.ParameterDict(
    {
        op.param_name: torch.rand(1, requires_grad=True)
        for op in ansatz
        if isinstance(op, Parametric)
    }
)
circ = QuantumCircuit(N_QUBITS, feature_map + ansatz).to(device=DEVICE, dtype=COMPLEX_DTYPE)
observable = TotalMagnetization(N_QUBITS).to(device=DEVICE, dtype=COMPLEX_DTYPE)
param_dict = param_dict.to(device=DEVICE, dtype=REAL_DTYPE)
state = circ.init_state()


def exp_fn(inputs: Tensor) -> Tensor:
    return expectation(
        circ,
        state,
        {**param_dict, **{VARIABLES[X_POS]: inputs[:, X_POS], VARIABLES[Y_POS]: inputs[:, Y_POS]}},
        observable,
        DIFF_MODE,
    )


single_domain_torch = linspace(0, 1, steps=N_POINTS)
domain_torch = tensor(list(product(single_domain_torch, single_domain_torch)))

opt = optim.Adam(param_dict.values(), lr=LEARNING_RATE)
sol = DomainSampling(exp_fn, len(VARIABLES), N_POINTS, DEVICE, REAL_DTYPE)

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
