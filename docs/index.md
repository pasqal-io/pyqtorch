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

hamiltonian_evolution = HamiltonianEvolution(qubit_support=[i for i in range(n_qubits)])

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

## Digital noisy simulation

In the description of closed quantum systems, a pure state vector is used to represent the complete quantum state. Thus, pure quantum states are represented by state vectors $\ket{\psi}$.

However, this description is not sufficient to study open quantum systems. When the system interacts with its environment, quantum systems can be in a mixed state, where quantum information is no longer entirely contained in a single state vector but is distributed probabilistically.

To address these more general cases, we consider a probabilistic combination $p_i$ of possible pure states $\ket{\psi_i}$. Thus, the system is described by a density matrix $\rho$ defined as follows:

$$
\rho = \sum_i p_i |\psi_i\rangle \langle \psi_i|
$$

The transformations of the density operator of an open quantum system interacting with its environment (noise) are represented by the super-operator $S: \rho \rightarrow S(\rho)$, often referred to as a quantum channel.
Quantum channels, due to the conservation of the probability distribution, must be CPTP (Completely Positive and Trace Preserving). Any CPTP super-operator can be written in the following form:

$$
S(\rho) = \sum_i K_i \rho K^{\dagger}_i
$$

Where $K_i$ are the Kraus operators, and satisfy the property $\sum_i K_i K^{\dagger}_i = \mathbb{I}$. As noise is the result of system interactions with its environment, it is therefore possible to simulate noisy quantum circuit with noise type gates.

Thus, `pyqtorch` implements a large selection of single qubit noise gates such as:

* The bit flip channel defined as:
    $$
        \textbf{BitFlip}(\rho) =(1-p) \rho + p X \rho X^{\dagger}
    $$
* The phase flip channel defined as:
    $$
        \textbf{PhaseFlip}(\rho) = (1-p) \rho + p Z \rho Z^{\dagger}
    $$
* The depolarizing channel defined as:
    $$
        \textbf{Depolarizing}(\rho) = (1-p) \rho + \frac{p}{3} (X \rho X^{\dagger}
            + Y \rho Y^{\dagger}
            + Z \rho Z^{\dagger})
    $$
* The pauli channel defined as:
    $$
        \textbf{PauliChannel}(\rho) = (1-p_x-p_y-p_z) \rho
            + p_x X \rho X^{\dagger}
            + p_y Y \rho Y^{\dagger}
            + p_z Z \rho Z^{\dagger}
    $$
* The amplitude damping channel defined as:
    $$
        \textbf{AmplitudeDamping}(\rho) =  K_0 \rho K_0^{\dagger} + K_1 \rho K_1^{\dagger}
    $$
    with:
    $\begin{equation*}
    K_{0} \ =\begin{pmatrix}
    1 & 0\\
    0 & \sqrt{1-\ \gamma }
    \end{pmatrix} ,\ K_{1} \ =\begin{pmatrix}
    0 & \sqrt{\ \gamma }\\
    0 & 0
    \end{pmatrix}
    \end{equation*}$
* The phase damping channel defined as:
    $$
        \textbf{PhaseDamping}(\rho) = K_0 \rho K_0^{\dagger} + K_1 \rho K_1^{\dagger}
    $$
    with:
    $\begin{equation*}
    K_{0} \ =\begin{pmatrix}
    1 & 0\\
    0 & \sqrt{1-\ \gamma }
    \end{pmatrix}, \ K_{1} \ =\begin{pmatrix}
    0 & 0\\
    0 & \sqrt{\ \gamma }
    \end{pmatrix}
    \end{equation*}$
* The generalize amplitude damping channel is defined as:
    $$
        \textbf{GeneralizedAmplitudeDamping}(\rho) = K_0 \rho K_0^{\dagger} + K_1 \rho K_1^{\dagger}
            + K_2 \rho K_2^{\dagger} + K_3 \rho K_3^{\dagger}
    $$
    with:
$\begin{cases}
K_{0} \ =\sqrt{p} \ \begin{pmatrix}
1 & 0\\
0 & \sqrt{1-\ \gamma }
\end{pmatrix} ,\ K_{1} \ =\sqrt{p} \ \begin{pmatrix}
0 & 0\\
0 & \sqrt{\ \gamma }
\end{pmatrix} \\
K_{2} \ =\sqrt{1\ -p} \ \begin{pmatrix}
\sqrt{1-\ \gamma } & 0\\
0 & 1
\end{pmatrix} ,\ K_{3} \ =\sqrt{1-p} \ \begin{pmatrix}
0 & 0\\
\sqrt{\ \gamma } & 0
\end{pmatrix}
\end{cases}$

 Noise gates are `Primitive` types, but they also request a `probability` argument to represent the noise affecting the system. And either a vector or a density matrix can be used as an input, but the output will always be a density matrix.

```python exec="on" source="material-block"
import torch
from pyqtorch.noise import AmplitudeDamping, PhaseFlip
from pyqtorch.utils import random_state

input_state = random_state(n_qubits=2)
noise_prob = 0.3
AmpD = AmplitudeDamping(0,noise_prob)
output_state = AmpD(input_state) #It's a density matrix
pf = PhaseFlip(1,0.7)
output_state = pf(output_state)
```

Noisy circuit initialization is the same as noiseless ones and the output will always be a density matrix. Let’s show its usage through the simulation of a realistic $X$ gate.

We know that an $X$ gate flips the state of the qubit, for instance $X|0\rangle = |1\rangle$. In practice, it's common for the target qubit to stay in its original state after applying $X$ due to the interactions between it and its environment. The possibility of failure can be represented by a `BitFlip` gate, which flips the state again after the application of the $X$ gate, returning it to its original state with a probability `1 - gate_fidelity`.

```python exec="on" source="material-block"
import matplotlib.pyplot as plt
import torch

from pyqtorch.circuit import QuantumCircuit
from pyqtorch.noise import Bitflip
from pyqtorch.primitive import X
from pyqtorch.utils import product_state


input_state = product_state('00')
x = X(0)
gate_fidelity = 0.9
bf = BitFlip(0,1.-gate_fidelity)
circ = QuantumCircuit(2,[x,bf])
output_state = circ(input_state)
output_state_diag = output_state.diagonal(dim1=0).real

plt.figure()
diag_values = output_state_diag.squeeze().numpy()
plt.bar(range(len(diag_values)), diag_values, color='blue', alpha=0.7)
custom_labels = ['00', '01', '10', '11']
plt.xticks(range(len(diag_values)), custom_labels)
plt.title("Probability of state occurrence")
plt.xlabel('Possible States')
plt.ylabel('Probability')
from io import StringIO  # markdown-exec: hide
from matplotlib.figure import Figure  # markdown-exec: hide
def fig_to_html(fig: Figure) -> str:  # markdown-exec: hide
    buffer = StringIO()  # markdown-exec: hide
    fig.savefig(buffer, format="svg")  # markdown-exec: hide
    return buffer.getvalue()  # markdown-exec: hide
print(fig_to_html(plt.gcf())) # markdown-exec: hide
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
from pyqtorch.circuit import hea
from pyqtorch.utils import DiffMode
from pyqtorch.parametric import Parametric
import matplotlib.pyplot as plt

from torch.nn.functional import mse_loss

# We can train on GPU if available
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# We can also choose the precision we want to train on
COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32
N_QUBITS = 4
DEPTH = 2
LR = .2
DIFF_MODE = DiffMode.ADJOINT
N_EPOCHS = 75

# Target function and some training data
fn = lambda x, degree: .05 * reduce(add, (torch.cos(i*x) + torch.sin(i*x) for i in range(degree)), 0)
x = torch.linspace(0, 10, 100)
y = fn(x, 5)
# Lets define a feature map to encode our 'x' values
feature_map = [pyq.RX(i, f'x') for i in range(N_QUBITS)]
# To fit the function, we define a hardware-efficient ansatz with tunable parameters
ansatz, params = hea(N_QUBITS, DEPTH, 'theta')
# Lets move all necessary components to the DEVICE
circ = pyq.QuantumCircuit(N_QUBITS, feature_map + ansatz).to(device=DEVICE, dtype=COMPLEX_DTYPE)
observable = pyq.Hamiltonian([pyq.Z(0)]).to(device=DEVICE, dtype=COMPLEX_DTYPE)
params = params.to(device=DEVICE, dtype=REAL_DTYPE)
x, y = x.to(device=DEVICE, dtype=REAL_DTYPE), y.to(device=DEVICE, dtype=REAL_DTYPE)
state = circ.init_state()

def exp_fn(params: dict[str, torch.Tensor], inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return pyq.expectation(circ, state, {**params,**inputs}, observable, DIFF_MODE)

with torch.no_grad():
    y_init = exp_fn(params, {'x': x})

# We need to set 'foreach' False since Adam doesnt support float64 on CUDA devices
optimizer = torch.optim.Adam(params.values(), lr=LR, foreach=False)

for _ in range(N_EPOCHS):
    optimizer.zero_grad()
    y_pred = exp_fn(params, {'x': x})
    loss = mse_loss(y, y_pred)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    y_final = exp_fn(params, {'x': x})

plt.figure()
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
from pyqtorch.circuit import hea
from pyqtorch import CNOT, RX, RY, QuantumCircuit, Z, expectation, Hamiltonian, Sequence, Merge
from pyqtorch.parametric import Parametric
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
total_magnetization = Hamiltonian([Z(i) for i in range(N_QUBITS)]).to(device=DEVICE, dtype=COMPLEX_DTYPE)
params = params.to(device=DEVICE, dtype=REAL_DTYPE)
state = circ.init_state()


def exp_fn(inputs: Tensor) -> Tensor:
    return expectation(
        circ,
        state,
        {**params, **{VARIABLES[X_POS]: inputs[:, X_POS], VARIABLES[Y_POS]: inputs[:, Y_POS]}},
        total_magnetization,
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
