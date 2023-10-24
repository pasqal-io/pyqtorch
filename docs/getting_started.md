## pyqtorch in a nutshell

Every gate in pyqtorch is a torch.nn.Module and can be instantiated using a target qubit and in the
case of a parametric gate, a param_name.

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
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import pyqtorch as pyq


# Let's define a target function we want to fit.

# In[2]:


def target_function(x, degree=3):
    result = 0
    for i in range(degree):
        result += torch.cos(i*x) + torch.sin(i*x)
    return .05 * result

x = torch.tensor(np.linspace(0, 10, 100))
target_y = target_function(x, 5)


def HEA(n_qubits, n_layers, param_name:str):
    ops = []
    for l in range(n_layers):
        ops.append(pyq.RX(i, f'{param_name}_0_{l}_{i}') for i in range(n_qubits))
        ops.append(pyq.RY(i, f'{param_name}_1_{l}_{i}') for i in range(n_qubits))
        ops.append(pyq.RX(i, f'{param_name}_2_{l}_{i}') for i in range(n_qubits))
        ops.append(pyq.CNOT(i % n_qubits, (i+1) % n_qubits) for i in range(n_qubits))
    return pyq.QuantumCircuit(n_qubits, ops)


class Model(pyq.QuantumCircuit):

    def __init__(self, n_qubits, n_layers):
        super().__init__(n_qubits, [])
        self.n_qubits = n_qubits
        self.ansatz1 = HEALayers(n_qubits, n_layers, 'theta')
        self.embedding = [pyq.RX(i, f'phi_{i}') for i in range(n_qubits)]
        self.ansatz2 = HEALayers(n_qubits, n_layers, 'epsilon')
        self.observable = pyq.Z(0)

    def forward(self, x):
        batch_size = len(x)
        state = self.ansatz1.init_state(batch_size)

        state = self.ansatz1(state)
        state = self.embedding(state, x)
        state = self.ansatz2(state)

        new_state = self.observable(state)

        return pyq.overlap(state,new_state)

n_qubits = 5
n_layers = 3
model = Model(n_qubits, n_layers)

with torch.no_grad():
    y = model(x)

plt.plot(x.numpy(), target_y.numpy(), label="truth")
plt.plot(x.numpy(), y.numpy(), label="initial")
plt.legend()
plt.show()

optimizer = torch.optim.Adam(model.parameters(), lr=.01)
epochs = 200

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = F.mse_loss(target_y, y_pred)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1:03d} | Loss {loss}")


with torch.no_grad():
    y_final = model(x)

plt.plot(x.numpy(), target_y.numpy(), label="truth")
plt.plot(x.numpy(), y.numpy(), label="initial")
plt.plot(x.numpy(), y_final.numpy(), "--", label="final", linewidth=3)
plt.legend()
plt.show()
```
