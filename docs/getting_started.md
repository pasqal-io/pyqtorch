## pyqtorch in a nutshell


```python exec="on" source="material-block"
import torch
import pyqtorch as pyq

x = pyq.X(0)
state = pyq.random_state(n_qubits=2)

print(x(state))

rx = pyq.RX(0, 'theta')
theta = torch.rand(1)
values = {'theta':theta}
print(rx(state, values ))

cnot = pyq.CNOT(0,1)
print(cnot(state))

crx = pyq.CRX(0, 1, 'theta')
print(crx(state,values))
```

## Fitting a function
```python exec="on" source="material-block" html="1"
from __future__ import annotations

import torch
import pyqtorch as pyq
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
        self.hea0 = HEA(n_qubits, n_layers, 'theta')
        self.embedding = pyq.QuantumCircuit(n_qubits, [pyq.RX(i, f'phi') for i in range(n_qubits)])
        self.hea1 = HEA(n_qubits, n_layers, 'epsilon')
        self.observable = pyq.Z(0)
        self.param_dict = torch.nn.ParameterDict({op.param_name: torch.rand(1, requires_grad=True) for op in self.hea0.operations if not isinstance(op, pyq.CNOT)})
        self.param_dict.update({op.param_name: torch.rand(1, requires_grad=True) for op in self.hea1.operations if not isinstance(op, pyq.CNOT)})
    def forward(self, phi: torch.Tensor):
        batch_size = len(phi)
        state = self.hea0.init_state(batch_size)
        state = self.hea0(state, self.param_dict)
        state = self.embedding(state, {'phi': phi})
        state = self.hea1(state, self.param_dict)
        new_state = self.observable(state)
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
    print(f"Epoch {epoch+1:03d} | Loss {loss}")


with torch.no_grad():
    y_final = model(x)

plt.plot(x.numpy(), y.numpy(), label="truth")
plt.plot(x.numpy(), y_init.numpy(), label="initial")
plt.plot(x.numpy(), y_final.numpy(), "--", label="final", linewidth=3)
plt.legend()
plt.show()
```
