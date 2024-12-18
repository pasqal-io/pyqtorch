## Digital Noise

In the description of closed quantum systems, a pure state vector is used to represent the complete quantum state. Thus, pure quantum states are represented by state vectors $|\psi \rangle $.

However, this description is not sufficient to study open quantum systems. When the system interacts with its environment, quantum systems can be in a mixed state, where quantum information is no longer entirely contained in a single state vector but is distributed probabilistically.

To address these more general cases, we consider a probabilistic combination $p_i$ of possible pure states $|\psi_i \rangle$. Thus, the system is described by a density matrix $\rho$ defined as follows:

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

- The bit flip channel defined as: $\textbf{BitFlip}(\rho) =(1-p) \rho + p X \rho X^{\dagger}$
- The phase flip channel defined as: $\textbf{PhaseFlip}(\rho) = (1-p) \rho + p Z \rho Z^{\dagger}$
- The depolarizing channel defined as: $\textbf{Depolarizing}(\rho) = (1-p) \rho + \frac{p}{3} (X \rho X^{\dagger} + Y \rho Y^{\dagger} + Z \rho Z^{\dagger})$
- The pauli channel defined as: $\textbf{PauliChannel}(\rho) = (1-p_x-p_y-p_z) \rho
            + p_x X \rho X^{\dagger}
            + p_y Y \rho Y^{\dagger}
            + p_z Z \rho Z^{\dagger}$
- The amplitude damping channel defined as: $\textbf{AmplitudeDamping}(\rho) =  K_0 \rho K_0^{\dagger} + K_1 \rho K_1^{\dagger}$
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
- The phase damping channel defined as: $\textbf{PhaseDamping}(\rho) = K_0 \rho K_0^{\dagger} + K_1 \rho K_1^{\dagger}$
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
* The generalize amplitude damping channel is defined as: $\textbf{GeneralizedAmplitudeDamping}(\rho) = K_0 \rho K_0^{\dagger} + K_1 \rho K_1^{\dagger} + K_2 \rho K_2^{\dagger} + K_3 \rho K_3^{\dagger}$
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

```python exec="on" source="material-block" html="1"
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
from pyqtorch.noise import BitFlip
from pyqtorch.primitives import X
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

## Analog Noise

Analog noise is made possible by specifying the `noise` argument in `HamiltonianEvolution` either as a list of tensors defining the jump operators to use when using Schrödinger or Lindblad equation solvers or a `AnalogNoise` instance. An `AnalogNoise` instance can be instantiated by providing a list of tensors, and a `qubit_support` that should be a subset of the qubit support of  `HamiltonianEvolution`.

```python exec="on" source="material-block"
import torch
from pyqtorch import uniform_state, HamiltonianEvolution
from pyqtorch.matrices import DEFAULT_MATRIX_DTYPE
from pyqtorch.noise import Depolarizing, AnalogNoise
from pyqtorch.utils import  SolverType

n_qubits = 2
qubit_targets = list(range(n_qubits))

# Random hermitian hamiltonian
matrix = torch.rand(2**n_qubits, 2**n_qubits, dtype=DEFAULT_MATRIX_DTYPE)
hermitian_matrix = matrix + matrix.T.conj()

time = torch.tensor([1.0])
time_symbol = "t"
dur_val = torch.rand(1)
noise_ops = Depolarizing(0, error_probability=0.1).tensor(2)
noise_ops = [op.squeeze() for op in noise_ops]
# also can be specified as AnalogNoise
noise_ops = AnalogNoise(noise_ops, qubit_support=(0,1))
solver = SolverType.DP5_ME
n_steps = 5

hamiltonian_evolution = HamiltonianEvolution(hermitian_matrix, time_symbol, qubit_targets,
        duration=dur_val, steps=n_steps,
        solver=solver, noise=noise_ops)

# Starting from a uniform state
psi_start = uniform_state(n_qubits)

# Returns the evolved state
psi_end = hamiltonian_evolution(state = psi_start, values={time_symbol: time})

print(psi_end) # markdown-exec: hide
```

There is one predefined `AnalogNoise` available: Depolarizing noise (`AnalogDepolarizing`) defined with jump operators: $L_{0,1,2} = \sqrt{\frac{p}{4}} (X, Y, Z)$. Note we can combine `AnalogNoise` with the `+` operator.



```python exec="on" source="material-block"
from pyqtorch.noise import AnalogDepolarizing
analog_noise = AnalogDepolarizing(error_param=0.1, qubit_support=0) + AnalogDepolarizing(error_param=0.1, qubit_support=1)
# we now have a qubit support acting on qubits 0 and 1
print(analog_noise.qubit_support) # markdown-exec: hide
```


## Readout errors

Another source of noise can be added when performing measurements. This is typically described using confusion matrices of the form:

$$
T(x|x')=\delta_{xx'}
$$

where $x$ represent a bitstring.
We provide two ways to define readout errors:
- `ReadoutNoise` : where each bit can be corrupted independently given an error probability or a 1D tensor of errors.
- `CorrelatedReadoutNoise` : where we provide the full confusion matrix for all possible bitstrings.

```python exec="on" source="material-block"
import torch
import pyqtorch as pyq
from pyqtorch.noise.readout import ReadoutNoise

rx = pyq.RX(0, param_name="theta")
y = pyq.Y(0)
cnot = pyq.CNOT(0, 1)
ops = [rx, y, cnot]
n_qubits = 2
circ = pyq.QuantumCircuit(n_qubits, ops)
state = pyq.random_state(n_qubits)
theta = torch.rand(1, requires_grad=True)
obs = pyq.Observable(pyq.Z(0))

noiseless_expectation = pyq.expectation(circ, state, {"theta": theta}, observable=obs)
readobj = ReadoutNoise(n_qubits, seed=0)
noisycirc = pyq.QuantumCircuit(n_qubits, ops, readobj)
noisy_expectation = pyq.expectation(noisycirc, state, {"theta": theta}, observable=obs, n_shots=1000)
print(f"Noiseless expectation: {noiseless_expectation.item()}") # markdown-exec: hide
print(f"Noisy expectation: {noisy_expectation.item()}") # markdown-exec: hide
```
