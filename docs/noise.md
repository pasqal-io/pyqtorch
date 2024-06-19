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
