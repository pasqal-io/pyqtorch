# Welcome to pyqtorch

**pyqtorch** is a [PyTorch](https://pytorch.org/)-based state vector simulator designed for quantum machine learning.

## Setup

To install `pyqtorch` , you can go into any virtual environment of your
choice and install it normally with `pip`:

```
pip install pyqtorch
```

## Digital

`pyqtorch` offers both primitive and parametric single to n-qubit, digital quantum gates.

```python exec="on" source="material-block"
import torch
import pyqtorch as pyq

x = pyq.X(0)
state = pyq.random_state(n_qubits=2)

x(state, None)

rx = pyq.RX(0, 'theta')
theta = torch.rand(1)
values = {'theta':theta}
rx(state, values )

cnot = pyq.CNOT(0,1)
cnot(state,None)

crx = pyq.CRX(0, 1, 'theta')
crx(state,values)
```

## Analog

`pyqtorch`s analog module also offers global state evolution through the `HamiltonianEvolution` class.

```python exec="on" source="material-block" html="1"
import torch
import pyqtorch as pyq


n_qubits = 4
sigmaz = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.cdouble))
Hbase = torch.kron(sigmaz, sigmaz)
hamiltonian = torch.kron(Hbase, Hbase)
t_evo = torch.tensor([torch.pi / 4], dtype=torch.cdouble)
hamevo = pyq.HamiltonianEvolution(hamiltonian=hamiltonian, time_evolution=t_evo, qubit_support=[i for i in range(n_qubits)], n_qubits=n_qubits)
psi = pyq.uniform_state(n_qubits)
psi_star = hamevo(psi)
result = pyq.overlap(psi_star, psi)
```


## QuantumCircuit

Digital gates can be wrapped in a `QuantumCircuit`, which are fully differentiable.

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
        state = self.hea.init_state(batch_size)
        state = self.feature_map(state, {'phi': phi})
        state = self.hea(state, self.param_dict)
        new_state = self.observable(state, self.param_dict)
        return pyq.overlap(state, new_state)

n_qubits = 5
n_layers = 3
model = QNN(n_qubits, n_layers)

with torch.no_grad():
    y_init = model(x)

plt.plot(x.numpy(), y.numpy(), label="truth")
plt.plot(x.numpy(), y_init.numpy(), label="initial")
plt.legend()
plt.show()

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
plt.show()
```
