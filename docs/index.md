# Welcome to pyqtorch

**pyqtorch** is a state vector simulator designed for quantum machine learning written in [PyTorch](https://pytorch.org/). It allows for building fully differentiable quantum circuits comprised of both digital and analog operations using a intuitive [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)-based API.

## Setup

To install `pyqtorch` , you can go into any virtual environment of your
choice and install it normally with `pip`:

```bash
pip install pyqtorch
```

## Digital Operations

`pyqtorch` implements a large selection of both primitive and parametric single to n-qubit, digital quantum gates.

Let's have a look at primitive gates first.

```python exec="on" source="material-block"
import torch
from pyqtorch import X, CNOT, random_state

x = X(0)
state = random_state(n_qubits=2)

new_state = x(state)

cnot = CNOT(0,1)
new_state= cnot(state)
```

Parametric gates can be initialized with or without a `param_name`. In the former case, a dictionary containing the `param_name` and a `torch.Tensor` for the parameter is expected when calling the forward method of the gate.

```python exec="on" source="material-block"
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


```python exec="on" source="material-block"
import torch
from pyqtorch import RX, random_state

state = random_state(n_qubits=2)
rx = RX(0)
new_state = rx(state, torch.rand(1))
```

## Analog Operations

`pyqtorch` also contains a `analog` module which allows for global state evolution through the `HamiltonianEvolution` class. Note that it only accepts a `torch.Tensor` as a generator which is expected to be an Hermitian matrix. To build arbitrary Pauli hamiltonians, we recommend using [Qadence](https://pasqal-io.github.io/qadence/v1.0.3/tutorials/hamiltonians/).

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

hamiltonian_evolution = HamiltonianEvolution(qubit_support=[i for i in range(n_qubits)])

# Starting from a uniform state
psi_start = uniform_state(n_qubits)

# Returns an evolved state at each time value
psi_end = hamiltonian_evolution(
    hamiltonian=hermitian_matrix,
    time_evolution=t_list,
    state = psi_start)

assert is_normalized(psi_end, atol=1e-05)
```

## Circuits

Using digital and analog operations, you can can build fully differentiable quantum circuits using the `QuantumCircuit` class; note that the default differentiation mode in pyqtorch is using torch.autograd.

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
obs = pyq.QuantumCircuit(n_qubits, [pyq.Z(0)])
expval = pyq.expectation(circ, state, {"theta": theta}, obs)
dfdtheta = torch.autograd.grad(expval, theta, torch.ones_like(expval))
```

## Adjoint Differentiation

`pyqtorch` also offers a [adjoint differentiation mode](https://arxiv.org/abs/2009.02823) which can be used through the `expectation` method.

```python exec="on" source="material-block"
import pyqtorch as pyq
import torch
from pyqtorch.utils import DiffMode

n_qubits = 3
batch_size = 1

rx = pyq.RX(0, param_name="x")
cnot = pyq.CNOT(1, 2)
ops = [rx, cnot]
n_qubits = 3
circ = pyq.QuantumCircuit(n_qubits, ops)

obs = pyq.QuantumCircuit(n_qubits, [pyq.Z(0)])
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

## Fitting a nonlinear function

Let's have a look at how the `QuantumCircuit` can be used to fit a function.

```python exec="on" source="material-block" html="1"
from __future__ import annotations

from operator import add
from functools import reduce
import torch
import pyqtorch as pyq
from pyqtorch.circuit import hea
from pyqtorch.utils import DiffMode
from pyqtorch.parametric import Parametric
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
observable = pyq.Hamiltonian([pyq.Z(0)]).to(device=DEVICE, dtype=COMPLEX_DTYPE)
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

## Fitting a partial differential equation using DQC

Finally, we show how to implement [DQC](https://arxiv.org/abs/2011.10395) to solve a partial differential equation using `pyqtorch`.

```python exec="on" source="material-block" html="1"
from __future__ import annotations

from functools import reduce
from itertools import product
from operator import add
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, exp, linspace, ones_like, optim, rand, sin, tensor
from torch.autograd import grad
from pyqtorch.circuit import hea
from pyqtorch import CNOT, RX, RY, QuantumCircuit, Z, expectation, Hamiltonian, Sequence, Merge
from pyqtorch.parametric import Parametric
from pyqtorch.utils import DiffMode

DIFF_MODE = DiffMode.AD
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# We can also choose the precision we want to train on
COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32
LR = .15
N_QUBITS = 4
DEPTH = 3
VARIABLES = ("x", "y")
N_VARIABLES = len(VARIABLES)
X_POS, Y_POS = [i for i in range(N_VARIABLES)]
BATCH_SIZE = 250
N_EPOCHS = 750


class DomainSampling(torch.nn.Module):
    def __init__(
        self, exp_fn: Callable[[Tensor], Tensor], n_inputs: int, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.exp_fn = exp_fn
        self.n_inputs = n_inputs
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

    def sample(self, requires_grad: bool = False) -> Tensor:
        return rand((self.batch_size, self.n_inputs), requires_grad=requires_grad, device=self.device, dtype=self.dtype)

    def left_boundary(self) -> Tensor:  # u(0,y)=0
        sample = self.sample()
        sample[:, X_POS] = 0.0
        return self.exp_fn(sample).pow(2).mean()

    def right_boundary(self) -> Tensor:  # u(L,y)=0
        sample = self.sample()
        sample[:, X_POS] = 1.0
        return self.exp_fn(sample).pow(2).mean()

    def top_boundary(self) -> Tensor:  # u(x,H)=0
        sample = self.sample()
        sample[:, Y_POS] = 1.0
        return self.exp_fn(sample).pow(2).mean()

    def bottom_boundary(self) -> Tensor:  # u(x,0)=f(x)
        sample = self.sample()
        sample[:, Y_POS] = 0.0
        return (self.exp_fn(sample) - sin(np.pi * sample[:, 0])).pow(2).mean()

    def interior(self) -> Tensor:  # uxx+uyy=0
        sample = self.sample(requires_grad=True)
        f = self.exp_fn(sample)
        dfdxy = grad(
            f,
            sample,
            ones_like(f),
            create_graph=True,
        )[0]
        dfdxxdyy = grad(
            dfdxy,
            sample,
            ones_like(dfdxy),
        )[0]

        return (dfdxxdyy[:, X_POS] + dfdxxdyy[:, Y_POS]).pow(2).mean()


feature_map = [RX(i, VARIABLES[X_POS]) for i in range(N_QUBITS // 2)] + [
    RX(i, VARIABLES[Y_POS]) for i in range(N_QUBITS // 2, N_QUBITS)
]
ansatz, params = hea(N_QUBITS, DEPTH, "theta")
circ = QuantumCircuit(N_QUBITS, feature_map + ansatz).to(device=DEVICE, dtype=COMPLEX_DTYPE)
total_magnetization = Hamiltonian([Z(i) for i in range(N_QUBITS)]).to(device=DEVICE, dtype=COMPLEX_DTYPE)
params = params.to(device=DEVICE, dtype=REAL_DTYPE)
state = circ.init_state()


def exp_fn(inputs: Tensor) -> Tensor:
    return expectation(
        circ,
        state,
        {**params, **{VARIABLES[X_POS]: inputs[:, X_POS], VARIABLES[Y_POS]: inputs[:, Y_POS]}},
        total_magnetization,
        DIFF_MODE,
    )


single_domain_torch = linspace(0, 1, steps=BATCH_SIZE)
domain_torch = tensor(list(product(single_domain_torch, single_domain_torch)))

opt = optim.Adam(params.values(), lr=LR)
sol = DomainSampling(exp_fn, len(VARIABLES), BATCH_SIZE, DEVICE, REAL_DTYPE)

for _ in range(N_EPOCHS):
    opt.zero_grad()
    loss = (
        sol.left_boundary()
        + sol.right_boundary()
        + sol.top_boundary()
        + sol.bottom_boundary()
        + sol.interior()
    )
    loss.backward()
    opt.step()

dqc_sol = exp_fn(domain_torch.to(DEVICE)).reshape(BATCH_SIZE, BATCH_SIZE).detach().cpu().numpy()
analytic_sol = (
    (exp(-np.pi * domain_torch[:, X_POS]) * sin(np.pi * domain_torch[:, Y_POS]))
    .reshape(BATCH_SIZE, BATCH_SIZE)
    .T
).numpy()


fig, ax = plt.subplots(1, 2, figsize=(7, 7))
ax[0].imshow(analytic_sol, cmap="turbo")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title("Analytical solution u(x,y)")
ax[1].imshow(dqc_sol, cmap="turbo")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_title("DQC solution")
from io import StringIO  # markdown-exec: hide
from matplotlib.figure import Figure  # markdown-exec: hide
def fig_to_html(fig: Figure) -> str:  # markdown-exec: hide
    buffer = StringIO()  # markdown-exec: hide
    fig.savefig(buffer, format="svg")  # markdown-exec: hide
    return buffer.getvalue()  # markdown-exec: hide
print(fig_to_html(plt.gcf())) # markdown-exec: hide
```

## Fitting a noisy sinusoid with quantum dropout

Here we will demonstrate an implemention [quantum dropout](https://arxiv.org/abs/2310.04120), for the case of fitting a noisy sine function.

```python exec="on" source="material-block" html="1"

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker

# from sklearn.preprocessing import MinMaxScaler
from torch import manual_seed, optim, tensor

import pyqtorch as pyq
from pyqtorch.circuit import DropoutQuantumCircuit
from pyqtorch.parametric import Parametric

manual_seed(12345)
np.random.seed(12345)


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min = None
        self.scale = None

    def fit(self, X):
        self.min = X.min(axis=0)
        self.scale = X.max(axis=0) - self.min
        self.scale[self.scale == 0] = 1  # Avoid division by zero for constant features

    def transform(self, X):
        X_scaled = (X - self.min) / self.scale
        X_scaled = (
            X_scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        )
        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        X = (X_scaled - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        X = X * self.scale + self.min
        return X


def sin_dataset(dataset_size=100, test_size=0.4, noise=0.4):
    x_ax = np.linspace(-1, 1, dataset_size)
    y = np.sin(x_ax * np.pi)
    noise = np.random.normal(0, 0.5, y.shape) * noise
    y += noise

    rng = np.random.default_rng(40)
    indices = rng.permutation(dataset_size)
    n_test = int(dataset_size * test_size)
    n_train = int(dataset_size * (1 - test_size))
    test_indices = indices[:n_test]
    train_indices = indices[n_test : (n_test + n_train)]
    x_train, x_test, y_train, y_test = (
        x_ax[train_indices],
        x_ax[test_indices],
        y[train_indices],
        y[test_indices],
    )

    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return x_train, x_test, y_train, y_test


def hea_ansatz(n_qubits, layer):
    ops = []
    for i in range(n_qubits):
        ops.append(pyq.RX(i, param_name=f"theta_{i}{layer}{0}"))

    for j in range(n_qubits - 1):
        ops.append(pyq.CNOT(control=j, target=j + 1))

    for i in range(n_qubits):
        ops.append(pyq.RZ(i, param_name=f"theta_{i}{layer}{1}"))

    for j in range(n_qubits - 1):
        ops.append(pyq.CNOT(control=j, target=j + 1))

    for i in range(n_qubits):
        ops.append(pyq.RX(i, param_name=f"theta_{i}{layer}{2}"))

    for j in range(n_qubits - 1):
        ops.append(pyq.CNOT(control=j, target=j + 1))
    return ops


def fm1(n_qubits):
    return [pyq.RY(i, "x1") for i in range(n_qubits)]


def fm2(n_qubits):
    return [pyq.RZ(i, "x2") for i in range(n_qubits)]


class QuantumModelBase(torch.nn.Module):
    def __init__(self, n_qubits, n_layers, device):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device

        self.embedding1 = fm1(n_qubits=n_qubits)
        self.embedding2 = fm2(n_qubits=n_qubits)

        self.params = torch.nn.ParameterDict()
        operations = self.build_operations()
        self.circuit = self.build_circuit(operations)
        self.observable = pyq.QuantumCircuit(n_qubits, [pyq.Z(i) for i in range(n_qubits)]).to(
            device=device, dtype=torch.complex64
        )

        self.params = self.params.to(device=device, dtype=torch.float32)

    def build_operations(self):
        operations = []
        for i in range(self.n_layers):
            operations += self.embedding1 + self.embedding2
            layer_i_ansatz = hea_ansatz(n_qubits=n_qubits, layer=i)
            operations += layer_i_ansatz
            for op in layer_i_ansatz:
                if isinstance(op, Parametric):
                    self.params[f"{op.param_name}"] = torch.randn(1, requires_grad=True)

        return operations

    def build_circuit(self, operations):
        return pyq.QuantumCircuit(
            n_qubits=self.n_qubits,
            operations=operations,
        ).to(device=self.device, dtype=torch.complex64)

    def forward(self, x):
        x = x.flatten()
        x_1 = {"x1": torch.asin(x)}
        x_2 = {"x2": torch.acos(x**2)}
        state = self.circuit.init_state(batch_size=int(x.shape[0]))

        out = pyq.expectation(
            circuit=self.circuit,
            state=state,
            values={**self.params, **x_1, **x_2},
            observable=self.observable,
        )

        return out


class DropoutModel(QuantumModelBase):
    def __init__(
        self, n_qubits, n_layers, device, dropout_mode="rotational_dropout", dropout_prob=0.03
    ):
        self.dropout_mode = dropout_mode
        self.dropout_prob = dropout_prob
        super().__init__(n_qubits, n_layers, device)

    def build_circuit(self, operations):
        return DropoutQuantumCircuit(
            n_qubits=self.n_qubits,
            operations=operations,
            dropout_mode=self.dropout_mode,
            dropout_prob=self.dropout_prob,
        ).to(device=self.device, dtype=torch.complex64)


def train_step(model, opt, data):
    opt.zero_grad()
    y_true = data[1].flatten()
    y_preds = model(data[0])
    loss = torch.nn.MSELoss()(y_preds, y_true)
    loss.backward()
    opt.step()

    return loss


def train(model, opt, x_train, y_train, x_test, y_test, epochs):
    train_loss_history = []
    validation_loss_history = []

    x_test = tensor(x_test).to(device, dtype=torch.float32)
    y_test = tensor(y_test).to(device, dtype=torch.float32).flatten()

    x_train = tensor(x_train).to(device, dtype=torch.float32)
    y_train = tensor(y_train).to(device, dtype=torch.float32).flatten()

    for epoch in range(epochs):
        model.train()

        train_loss = train_step(model, opt, (x_train, y_train))

        model.eval()
        test_preds = model(x_test)
        test_loss = torch.nn.MSELoss()(test_preds, y_test).detach().numpy()

        train_loss_history.append(train_loss.detach().numpy())
        validation_loss_history.append(test_loss)

        if epoch % 100 == 0:
            print(f"epoch: {epoch}, train loss {train_loss}, val loss: {test_loss}")

    return model, train_loss_history, validation_loss_history


def plot_results(train_history, test_history, epochs):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.05)
    ax[0].set_title("MSE train")
    for k, v in train_history.items():
        mean_train_history = np.mean(v, axis=0)
        std_train_history = np.std(v, axis=0)
        print(mean_train_history.shape)
        ax[0].fill_between(
            range(epochs),
            mean_train_history - std_train_history,
            mean_train_history + std_train_history,
            alpha=0.2,
        )
        # average trend
        ax[0].plot(range(epochs), mean_train_history, label=f"{k}")

    ax[1].set_title("MSE test")
    for k, v in test_history.items():
        mean_test_history = np.mean(v, axis=0)
        std_test_history = np.std(v, axis=0)
        ax[1].fill_between(
            range(epochs),
            mean_test_history - std_test_history,
            mean_test_history + std_test_history,
            alpha=0.2,
        )
        # average trend
        ax[1].plot(range(epochs), mean_test_history, label=f"{k}")

    ax[0].legend(
        loc="upper center", bbox_to_anchor=(1.01, 1.25), ncol=4, fancybox=True, shadow=True
    )

    for ax in ax.flat:
        ax.set_xlabel("Epochs")
        ax.set_ylabel("MSE")
        ax.set_yscale("log")
        ax.label_outer()

    plt.subplots_adjust(bottom=0.3)
    plt.savefig("results.pdf")
    plt.close()


device = torch.device("cpu")
n_qubits = 4
depth = 20
lr = 0.01
n_points = 20
epochs = 1000
dropout_prob = 0.01

x_train, x_test, y_train, y_test = sin_dataset(dataset_size=n_points, test_size=0.25)

# visualising the function we will train a model to learn
fig, ax = plt.subplots()
plt.plot(x_train, y_train, "o", label="training")
plt.plot(x_test, y_test, "o", label="testing")
plt.plot(
    np.linspace(-1, 1, 100),
    [np.sin(x * np.pi) for x in np.linspace(-1, 1, 100)],
    linestyle="dotted",
    label=r"$\sin(x)$",
)
plt.ylabel(r"$y = \sin(\pi\cdot x) + \epsilon$")
plt.xlabel(r"$x$")
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
plt.legend()
plt.savefig("problem_function.pdf")
plt.close()

# scale the data to ensure it lies in the range (-1,1)
scaler = MinMaxScaler(feature_range=(-1, 1))
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)


num_runs = 3
train_history = {}
test_history = {}

# training with no dropout
train_losses = []
test_losses = []
for _ in range(num_runs):
    # defining an overparameterised quantum circuit
    model = QuantumModelBase(n_qubits=n_qubits, n_layers=depth, device=device)
    opt = optim.Adam(model.parameters(), lr=lr)
    _, train_loss_hist, test_loss_hist = train(
        model=model,
        opt=opt,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=epochs,
    )
    train_losses.append(train_loss_hist)
    test_losses.append(test_loss_hist)

train_losses = np.array(train_losses)
val_losses = np.array(test_losses)
train_history["no dropout"] = train_losses
test_history["no dropout"] = test_losses


# training with dropout
train_losses = []
test_losses = []
for _ in range(num_runs):
    # defining a quantum model with dropout
    model = DropoutModel(
        n_qubits=n_qubits,
        n_layers=depth,
        device=device,
        dropout_mode="rotational_dropout",
        dropout_prob=dropout_prob,
    )

    opt = optim.Adam(model.parameters(), lr=lr)
    _, train_loss_hist, val_loss_hist = train(
        model=model,
        opt=opt,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=epochs,
    )
    train_losses.append(train_loss_hist)
    test_losses.append(test_loss_hist)

train_losses = np.array(train_losses)
val_losses = np.array(test_losses)
train_history["dropout"] = train_losses
test_history["dropout"] = test_losses


plot_results(train_history=train_history, test_history=test_history, epochs=epochs)


```
## CUDA Profiling and debugging

To debug your quantum programs on `CUDA` devices, `pyqtorch` offers a `DEBUG` mode, which can be activated via
setting the `PYQ_LOG_LEVEL` environment variable.

```bash
export PYQ_LOG_LEVEL=DEBUG
```

Before running your script, make sure to install the following packages:

```bash
pip install nvidia-pyindex
pip install nvidia-dlprof[pytorch]
```
For more information, check [the dlprof docs](https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/index.html).
