Let's have a look at how the `QuantumCircuit` can be used to fit a simple nonlienar function.

```python exec="on" source="material-block" html="1"
from __future__ import annotations

from operator import add
from functools import reduce
import torch
import pyqtorch as pyq
from pyqtorch.composite import hea
from pyqtorch.utils import DiffMode
from pyqtorch.primitives import Parametric
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
observable = pyq.Observable(pyq.Z(0)).to(device=DEVICE, dtype=COMPLEX_DTYPE)
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
