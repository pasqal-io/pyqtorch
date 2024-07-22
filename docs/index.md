# Welcome to pyqtorch

**pyqtorch** is a state vector simulator designed for quantum machine learning written in [PyTorch](https://pytorch.org/). It allows for building fully differentiable quantum circuits comprised of both digital and analog operations using a intuitive [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)-based API.
It can be used standalone as shown in these docs, or through our framework for quantum programming: [Qadence](https://github.com/pasqal-io/qadence).

## Setup

To install `pyqtorch` , you can go into any virtual environment of your
choice and install it normally with `pip`:

```bash
pip install pyqtorch
```

## Digital Operations

`pyqtorch` implements a large selection of both primitive and parametric single to n-qubit, digital quantum gates.

Let's have a look at primitive gates first.

```python exec="on" source="material-block" html="1"
import torch
from pyqtorch import X, CNOT, random_state

x = X(0)
state = random_state(n_qubits=2)

new_state = x(state)

cnot = CNOT(0,1)
new_state= cnot(state)
```

Parametric gates can be initialized with or without a `param_name`. In the former case, a dictionary containing the `param_name` and a `torch.Tensor` for the parameter is expected when calling the forward method of the gate.

```python exec="on" source="material-block" html="1"
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


```python exec="on" source="material-block" html="1"
import torch
from pyqtorch import RX, random_state

state = random_state(n_qubits=2)
rx = RX(0)
new_state = rx(state, torch.rand(1))
```

## Analog Operations

An analog operation is one whose unitary is best described by the evolution of some hermitian generator, or Hamiltonian, acting on an arbitrary number of qubits. For a time-independent generator $\mathcal{H}$ and some time variable $t$, the evolution operator is $\exp(-i\mathcal{H}t)$.

`pyqtorch` also contains a `analog` module which allows for global state evolution through the `HamiltonianEvolution` class. There exists several ways to pass a generator, and we present them in [Analog Operations](./analog.md). Below, we show an example where the generator $\mathcal{H}$ is an arbitrary tensor. To build arbitrary Pauli hamiltonians, we recommend using [Qadence](https://pasqal-io.github.io/qadence/v1.0.3/tutorials/hamiltonians/).

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

hamiltonian_evolution = HamiltonianEvolution(hermitian_matrix, t_list, [i for i in range(n_qubits)])

# Starting from a uniform state
psi_start = uniform_state(n_qubits)

# Returns an evolved state at each time value
psi_end = hamiltonian_evolution(
    state = psi_start)

assert is_normalized(psi_end, atol=1e-05)
```

!!! warning "Dimensionless units"
    The quantity $\mathcal{H}t$ has to be considered **dimensionless** for exponentiation in `pyqtorch`.

## Circuits

Using digital and analog operations, you can can build fully differentiable quantum circuits using the `QuantumCircuit` class; note that the default differentiation mode in pyqtorch is using torch.autograd.

```python exec="on" source="material-block" html="1"
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
obs = pyq.Observable(pyq.Z(0))
expval = pyq.expectation(circ, state, {"theta": theta}, obs)
dfdtheta = torch.autograd.grad(expval, theta, torch.ones_like(expval))
```
