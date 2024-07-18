## Analog Operations

An analog operation is one whose unitary is best described by the evolution of some hermitian generator, or Hamiltonian, acting on an arbitrary number of qubits. For a time-independent generator $\mathcal{H}$ and some time variable $t$, the evolution operator is $\exp(-i\mathcal{H}t)$. `pyqtorch` provides the HamiltonianEvolution class to initialize analog operations. There exists several ways to pass a generator, and we present them next.

!!! warning "Dimensionless units"
    The quantity $\mathcal{H}t$ has to be **dimensionless** for exponentiation in PyQTorch.

### Tensor generator

The first case of generator we can provide is simply an arbitrary hermitian tensor.
Note we can use a string for defining the time evolution as a parameter, instead of directly passing a tensor.

```python exec="on" source="material-block" html="1"
import torch
from pyqtorch import uniform_state, HamiltonianEvolution
from pyqtorch.matrices import DEFAULT_MATRIX_DTYPE

n_qubits = 2
qubit_targets = list(range(n_qubits))

# Random hermitian hamiltonian
matrix = torch.rand(2**n_qubits, 2**n_qubits, dtype=DEFAULT_MATRIX_DTYPE)
hermitian_matrix = matrix + matrix.T.conj()

time = torch.tensor([1.0])
time_symbol = "t"

hamiltonian_evolution = HamiltonianEvolution(hermitian_matrix, time_symbol, qubit_targets)

# Starting from a uniform state
psi_start = uniform_state(n_qubits)

# Returns the evolved state
psi_end = hamiltonian_evolution(state = psi_start, values={time_symbol: time})

print(psi_end)
```

### Symbol generator

We can also have a symbol generator to be replaced later by any hermitian matrix. and in this case we use a string symbol to instantiate `HamiltonianEvolution`.

```python exec="on" source="material-block" html="1"
import torch
from pyqtorch import uniform_state, HamiltonianEvolution
from pyqtorch.matrices import DEFAULT_MATRIX_DTYPE

n_qubits = 2
qubit_targets = list(range(n_qubits))

# Symbol hamiltonian
hermitian_symbol = "h"

time = torch.tensor([1.0])

hamiltonian_evolution = HamiltonianEvolution(hermitian_symbol, time, qubit_targets)

# Starting from a uniform state
psi_start = uniform_state(n_qubits)

# Set the value for h
matrix = torch.rand(2**n_qubits, 2**n_qubits, dtype=DEFAULT_MATRIX_DTYPE)
H = matrix + matrix.T.conj()

# Returns the evolved state
psi_end = hamiltonian_evolution(state = psi_start, values={hermitian_symbol: H})

print(psi_end)
```

### Sequence generator

The generator can be also a sequence of operators such as a quantum circuit:

```python exec="on" source="material-block" html="1"
import torch
from pyqtorch import uniform_state, HamiltonianEvolution, X, Y
from pyqtorch import Add, QuantumCircuit

n_qubits = 2

ops = [X, Y] * 2
qubit_targets = list(range(n_qubits))
generator = QuantumCircuit(
    n_qubits,
    [
        Add([op(q) for op, q in zip(ops, qubit_targets)]),
        *[op(q) for op, q in zip(ops, qubit_targets)],
    ],
)

time = torch.tensor([1.0])

hamiltonian_evolution = HamiltonianEvolution(generator, time, qubit_targets)

# Starting from a uniform state
psi_start = uniform_state(n_qubits)

# Returns the evolved state
psi_end = hamiltonian_evolution(state = psi_start)

print(psi_end)
```

### Batched execution

We also allow for different ways to run analog operations on batched inputs. We can have batched evolution times, or batched generators.
Below we show a few examples.

#### Batched evolution times


```python exec="on" source="material-block" html="1"
import torch
from pyqtorch import uniform_state, HamiltonianEvolution
from pyqtorch.matrices import DEFAULT_MATRIX_DTYPE

n_qubits = 2
qubit_targets = list(range(n_qubits))

# Random hermitian hamiltonian
matrix = torch.rand(2**n_qubits, 2**n_qubits, dtype=DEFAULT_MATRIX_DTYPE)
hermitian_matrix = matrix + matrix.T.conj()

times = torch.tensor([0.25, 0.5, 0.75, 1.0])
time_symbol = "t"

hamiltonian_evolution = HamiltonianEvolution(hermitian_matrix, time_symbol, qubit_targets)

# Starting from a uniform state
psi_start = uniform_state(n_qubits)

# Returns the evolved state
psi_end = hamiltonian_evolution(state = psi_start, values={time_symbol: times})

print(psi_end.size())
```

#### Batched generators



```python exec="on" source="material-block" html="1"
import torch
from pyqtorch import uniform_state, HamiltonianEvolution
from pyqtorch.matrices import DEFAULT_MATRIX_DTYPE

n_qubits = 2
qubit_targets = list(range(n_qubits))

# Random hermitian hamiltonian
matrix = torch.rand(2**n_qubits, 2**n_qubits, dtype=DEFAULT_MATRIX_DTYPE)
H = matrix + matrix.T.conj()
hermitian_batch = torch.stack((H, H.conj()), dim=2)

time = torch.tensor([1.0])
time_symbol = "t"

hamiltonian_evolution = HamiltonianEvolution(hermitian_batch, time_symbol, qubit_targets)

# Starting from a uniform state
psi_start = uniform_state(n_qubits)

# Returns the evolved state
psi_end = hamiltonian_evolution(state = psi_start, values={time_symbol: time})

print(psi_end.size())
```
