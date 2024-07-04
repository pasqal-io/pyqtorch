`pyqtorch` also offers several differentiation modes to compute gradients which can be accessed through the
`expectation` API. Simply pass one of three `DiffMode` options to the `diff_mode` argument.
The default is `ad`.

### Automatic Differentiation (DiffMode.AD)
The default differentation mode of `pyqtorch`, [torch.autograd](https://pytorch.org/docs/stable/autograd.html).
It uses the `torch` native automatic differentiation engine which tracks operations on `torch.Tensor` objects by constructing a computational graph to perform chain rules for derivatives calculations.

### Adjoint Differentiation (DiffMode.ADJOINT)
The [adjoint differentiation mode](https://arxiv.org/abs/2009.02823) computes first-order gradients by only requiring at most three states in memory in `O(P)` time where `P` is the number of parameters in a circuit.

### Generalized Parameter-Shift rules (DiffMode.GPSR)
To be added.

### Example
```python exec="on" source="material-block" html="1"
import pyqtorch as pyq
import torch
from pyqtorch.utils import DiffMode

n_qubits = 3
batch_size = 1

ry = pyq.RY(0, param_name="x")
cnot = pyq.CNOT(1, 2)
ops = [ry, cnot]
n_qubits = 3
circ = pyq.QuantumCircuit(n_qubits, ops)

obs = pyq.Observable(n_qubits, [pyq.Z(0)])
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
