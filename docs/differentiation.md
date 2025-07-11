## Differentiation

`pyqtorch` also offers several differentiation modes to compute gradients which can be accessed through the
`expectation` API. Simply pass one of three `DiffMode` options to the `diff_mode` argument.
The default is `ad`.

### Automatic Differentiation (DiffMode.AD)
The default differentation mode of `pyqtorch`, [torch.autograd](https://pytorch.org/docs/stable/autograd.html).
It uses the `torch` native automatic differentiation engine which tracks operations on `torch.Tensor` objects by constructing a computational graph to perform chain rules for derivatives calculations.

### Adjoint Differentiation (DiffMode.ADJOINT)
The [adjoint differentiation mode](https://arxiv.org/abs/2009.02823) computes first-order gradients by only requiring at most three states in memory in `O(P)` time where `P` is the number of parameters in a circuit.

### Generalized Parameter-Shift rules (DiffMode.GPSR)
The Generalized parameter shift rule (GPSR mode) is an extension of the well known [parameter shift rule (PSR)](https://arxiv.org/abs/1811.11184) algorithm [to arbitrary quantum operations](https://arxiv.org/abs/2108.01218). Indeed, PSR only works for quantum operations whose generator has a single gap in its eigenvalue spectrum, GPSR extending to multi-gap.

!!! warning "Usage restrictions"
    At the moment, circuits with one or more `Scale` or `HamiltonianEvolution` with parametric generators operations are not supported.
    They should be handled differently as GPSR requires operations to be of the form presented below.

For this, we define the differentiable function as quantum expectation value

$$
f(x) = \left\langle 0\right|\hat{U}^{\dagger}(x)\hat{C}\hat{U}(x)\left|0\right\rangle
$$

where $\hat{U}(x)={\rm exp}{\left( -i\frac{x}{2}\hat{G}\right)}$ is the quantum evolution operator
with generator $\hat{G}$ representing the structure of the underlying quantum circuit and $\hat{C}$ is the cost operator.
Then using the eigenvalue spectrum $\lambda_n$ of the generator $\hat{G}$
we calculate the full set of corresponding unique non-zero spectral gaps ${ \Delta_s\}$ (differences between eigenvalues).
It can be shown that the final expression of derivative of $f(x)$ is then given by the following expression:

$$
\begin{equation}
\frac{{\rm d}f\left(x\right)}{{\rm d}x}=\overset{S}{\underset{s=1}{\sum}}\Delta_{s}R_{s},
\end{equation}
$$

where $S$ is the number of unique non-zero spectral gaps and $R_s$ are real quantities that are solutions of a system of linear equations

$$
\begin{equation}
\begin{cases}
F_{1} & =4\overset{S}{\underset{s=1}{\sum}}{\rm sin}\left(\frac{\delta_{1}\Delta_{s}}{2}\right)R_{s},\\
F_{2} & =4\overset{S}{\underset{s=1}{\sum}}{\rm sin}\left(\frac{\delta_{2}\Delta_{s}}{2}\right)R_{s},\\
 & ...\\
F_{S} & =4\overset{S}{\underset{s=1}{\sum}}{\rm sin}\left(\frac{\delta_{M}\Delta_{s}}{2}\right)R_{s}.
\end{cases}
\end{equation}
$$

Here $F_s=f(x+\delta_s)-f(x-\delta_s)$ denotes the difference between values of functions evaluated at shifted arguments $x\pm\delta_s$.

!!! caution "Using GPSR with HamiltonianEvolution"
    GPSR works with the formalism above-presented, which corresponds to many parametric operations such as rotation gates.
    For HamiltonianEvolution, since the factor 1/2 is missing, to allow GPSR differentiation, we multiply by 2 the
    spectral gaps. Also we use a shift prefactor of 0.5 for multi-gap GPSR or 0.5 divided by the spectral gap for single-gap GPSR.


## Examples

### Circuit parameters differentiation

We show below a code example with several differentiation methods for circuit parameters:

```python exec="on" source="material-block" html="1" session="diff"
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

obs = pyq.Observable(pyq.Z(0))
state = pyq.zero_state(n_qubits)

values_ad = {"x": torch.tensor([torch.pi / 2], requires_grad=True)}
values_adjoint = {"x": torch.tensor([torch.pi / 2], requires_grad=True)}
values_gpsr = {"x": torch.tensor([torch.pi / 2], requires_grad=True)}

exp_ad = pyq.expectation(circ, state, values_ad, obs, DiffMode.AD)
exp_adjoint = pyq.expectation(circ, state, values_adjoint, obs, DiffMode.ADJOINT)
exp_gpsr = pyq.expectation(circ, state, values_gpsr, obs, DiffMode.GPSR)

dfdx_ad = torch.autograd.grad(exp_ad, tuple(values_ad.values()), torch.ones_like(exp_ad))

dfdx_adjoint = torch.autograd.grad(
    exp_adjoint, tuple(values_adjoint.values()), torch.ones_like(exp_adjoint)
)

dfdx_gpsr = torch.autograd.grad(
    exp_gpsr, tuple(values_gpsr.values()), torch.ones_like(exp_gpsr)
)

assert len(dfdx_ad) == len(dfdx_adjoint) == len(dfdx_gpsr)
for i in range(len(dfdx_ad)):
    assert torch.allclose(dfdx_ad[i], dfdx_adjoint[i])
    assert torch.allclose(dfdx_ad[i], dfdx_gpsr[i])
```

### Parametrized observable differentiation

To allow differentiating observable parameters only, we need to specify the `values` argument as a dictionary with two keys `circuit` and `observables`, each being a dictionary of corresponding parameters and values, as follows:

```python exec="on" source="material-block" html="1" session="diff"

obs = pyq.Observable(pyq.RZ(0, "obs"))
values_obs = {"obs": torch.tensor([torch.pi / 2], requires_grad=True)}
values = {"circuit": values_ad, "observables": values_obs}
exp_ad_separate = pyq.expectation(circ, state, values, obs, DiffMode.AD)
grad_ad_obs = torch.autograd.grad(
    exp_ad_separate, tuple(values_obs.values()), torch.ones_like(exp_ad_separate)
)
assert len(grad_ad_obs) == len(obs.params)
```
