# Welcome to pyqtorch

**pyqtorch** is a state vector simulator designed for quantum machine learning written in [PyTorch](https://pytorch.org/). It allows for building fully differentiable quantum circuits comprised of both digital and analog operations using a intuitive [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)-based API.

## Setup

To install `pyqtorch` , you can go into any virtual environment of your
choice and install it normally with `pip`:

```bash
pip install pyqtorch
```

## Digital

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

## Analog

`pyqtorch` also contains a `analog` module which allows for global state evolution through the `HamiltonianEvolution` class. Note that it only accepts a `torch.Tensor` as a generator which is expected to be an Hermitian matrix. To build arbitrary Pauli hamiltonians, we recommend using [Qadence](https://pasqal-io.github.io/qadence/v1.0.3/tutorials/hamiltonians/).

```python exec="on" source="material-block" html="1"
import torch
from pyqtorch import uniform_state, HamiltonianEvolution, is_normalized

n_qubits = 4

# Random hermitian hamiltonian
matrix = torch.rand(2**n_qubits, 2**n_qubits, dtype=torch.cdouble)
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

assert is_normalized(psi_end, atol = 1e-12)
```

## QuantumCircuit

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

def _fwd(phi: torch.Tensor) -> torch.Tensor:
    return circ(state, {"theta": theta})

assert torch.autograd.gradcheck(_fwd, theta)
```

## Fitting a function

Let's have a look at how the `QuantumCircuit` can be used to implement a Quantum Neural Network and fit a simple function.

```python exec="on" source="material-block" html="1"
from __future__ import annotations

import torch
import pyqtorch as pyq
from pyqtorch.parametric import Parametric
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F

def target_function(x: torch.Tensor, degree: int = 3) -> torch.Tensor:
    result = 0
    for i in range(degree):
        result += torch.cos(i*x) + torch.sin(i*x)
    return .05 * result

x = torch.tensor(np.linspace(0, 10, 100))
y = target_function(x, 5)


def HEA(n_qubits: int, n_layers: int, param_name: str) -> pyq.QuantumCircuit:
    ops = []
    for l in range(n_layers):
        ops += [pyq.RX(i, f'{param_name}_0_{l}_{i}') for i in range(n_qubits)]
        ops += [pyq.RY(i, f'{param_name}_1_{l}_{i}') for i in range(n_qubits)]
        ops += [pyq.RX(i, f'{param_name}_2_{l}_{i}') for i in range(n_qubits)]
        ops += [pyq.CNOT(i % n_qubits, (i+1) % n_qubits) for i in range(n_qubits)]
    return pyq.QuantumCircuit(n_qubits, ops)


class QNN(pyq.QuantumCircuit):

    def __init__(self, n_qubits, n_layers):
        super().__init__(n_qubits, [])
        self.n_qubits = n_qubits
        self.feature_map = pyq.QuantumCircuit(n_qubits, [pyq.RX(i, f'phi') for i in range(n_qubits)])
        self.hea = HEA(n_qubits, n_layers, 'theta')
        self.observable = pyq.Z(0)
        self.param_dict = torch.nn.ParameterDict({op.param_name: torch.rand(1, requires_grad=True) for op in self.hea.operations if isinstance(op, Parametric)})
    def forward(self, phi: torch.Tensor):
        batch_size = len(phi)
        state = self.feature_map.init_state(batch_size)
        state = self.feature_map(state, {'phi': phi})
        state = self.hea(state, self.param_dict)
        new_state = self.observable(state, self.param_dict)
        return pyq.overlap(state, new_state)

n_qubits = 5
n_layers = 3
model = QNN(n_qubits, n_layers)

with torch.no_grad():
    y_init = model(x)

optimizer = torch.optim.Adam(model.parameters(), lr=.01)
epochs = 200

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = F.mse_loss(y, y_pred)
    loss.backward()
    optimizer.step()


with torch.no_grad():
    y_final = model(x)

plt.plot(x.numpy(), y.numpy(), label="truth")
plt.plot(x.numpy(), y_init.numpy(), label="initial")
plt.plot(x.numpy(), y_final.numpy(), "--", label="final", linewidth=3)
plt.legend()
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```

## First Order Adjoint Differentiation

`pyqtorch` also offers a [adjoint differentiation mode](https://arxiv.org/abs/2009.02823) which can be used through the `expectation` method of `QuantumCircuit`.

```python exec="on" source="material-block"
import pyqtorch as pyq
import torch
from pyqtorch.utils import DiffMode

n_qubits = 3
batch_size = 1
diff_mode = DiffMode.ADJOINT


rx = pyq.RX(0, param_name="theta_0")
cry = pyq.CPHASE(0, 1, param_name="theta_1")
rz = pyq.RZ(2, param_name="theta_2")
cnot = pyq.CNOT(1, 2)
ops = [rx, cry, rz, cnot]
n_qubits = 3
adjoint_circ = pyq.QuantumCircuit(n_qubits, ops, DiffMode.ADJOINT)
ad_circ = pyq.QuantumCircuit(n_qubits, ops, DiffMode.AD)
obs = pyq.QuantumCircuit(n_qubits, [pyq.Z(0)])

theta_0_value = torch.pi / 2
theta_1_value = torch.pi
theta_2_value = torch.pi / 4

state = pyq.zero_state(n_qubits)

theta_0_ad = torch.tensor([theta_0_value], requires_grad=True)
thetas_0_adjoint = torch.tensor([theta_0_value], requires_grad=True)

theta_1_ad = torch.tensor([theta_1_value], requires_grad=True)
thetas_1_adjoint = torch.tensor([theta_1_value], requires_grad=True)

theta_2_ad = torch.tensor([theta_2_value], requires_grad=True)
thetas_2_adjoint = torch.tensor([theta_2_value], requires_grad=True)

values_ad = {"theta_0": theta_0_ad, "theta_1": theta_1_ad, "theta_2": theta_2_ad}
values_adjoint = {
    "theta_0": thetas_0_adjoint,
    "theta_1": thetas_1_adjoint,
    "theta_2": thetas_2_adjoint,
}
exp_ad = ad_circ.expectation(values_ad, obs, state)
exp_adjoint = adjoint_circ.expectation(values_adjoint, obs, state)

grad_ad = torch.autograd.grad(exp_ad, tuple(values_ad.values()), torch.ones_like(exp_ad))

grad_adjoint = torch.autograd.grad(
    exp_adjoint, tuple(values_adjoint.values()), torch.ones_like(exp_adjoint)
)

assert len(grad_ad) == len(grad_adjoint)
for i in range(len(grad_ad)):
    assert torch.allclose(grad_ad[i], grad_adjoint[i])
```
