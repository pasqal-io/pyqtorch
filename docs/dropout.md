## Fitting a noisy sinusoid with quantum dropout

Here we will demonstrate an implemention [quantum dropout](https://arxiv.org/abs/2310.04120), for the case of fitting a noisy sine function. To show the usefulness of dropout for quantum neural networks (QNNs), we shall compare the performance of a QNN with dropout and one without.

Firstly, we define the dataset that we will perform regression on, this function is $sin(\pi x)+\epsilon$, where $x\in\reals$ and $\epsilon$ is noise sampled from a normal distribution which is then added to each point.

```python exec="on" source="material-block" session="dropout"
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import torch
from torch import manual_seed, optim, tensor

import pyqtorch as pyq
from pyqtorch.circuit import DropoutQuantumCircuit
from pyqtorch.primitives import Parametric
from pyqtorch.utils import DropoutMode

seed = 70
manual_seed(seed)
np.random.seed(seed)

# choose device and hyperparameters

device = torch.device("cpu")
n_qubits = 2 # a greater performance difference is observed with 5 or more qubits
depth = 5 # a greater performance difference is observed at depth 10
lr = 0.01
n_points = 20
epochs = 100 # a greater performance difference is observed at 200-250 epochs of training
dropout_prob = 0.06
noise = 0.4

def sin_dataset(dataset_size: int = 100, test_size: float = 0.4, noise: float = 0.4):
    """Generates points (x,y) which follow sin(πx)+ϵ,
        where epsilon is noise randomly sampled from the normal
        distribution for each datapoint.
    Args:
        dataset_size (int): total number of points. Defaults to 100.
        test_size (float): fraction of points for testing. Defaults to 0.4.
        noise (float): standard deviation of added noise. Defaults to 0.4.
    Returns:
        data (tuple): data divided into training and testing
    """
    x_ax = np.linspace(-1, 1, dataset_size)
    y = np.sin(x_ax * np.pi)
    noise = np.random.normal(0, 0.5, y.shape) * noise
    y += noise

    # permuting the points around before dividing into train and test sets.
    rng = np.random.default_rng(seed)
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


# generates points following sin(πx)+ϵ, split into training and testing sets.
x_train, x_test, y_train, y_test = sin_dataset(dataset_size=n_points, test_size=0.25, noise=0.4)
```

We can now visualise the function we will train QNNs to learn.

```python exec="on" source="material-block" html="1" session="dropout"
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
from io import StringIO  # markdown-exec: hide
from matplotlib.figure import Figure  # markdown-exec: hide
def fig_to_html(fig: Figure) -> str:  # markdown-exec: hide
    buffer = StringIO()  # markdown-exec: hide
    fig.savefig(buffer, format="svg")  # markdown-exec: hide
    return buffer.getvalue()  # markdown-exec: hide
print(fig_to_html(plt.gcf())) # markdown-exec: hide
```

Since our QNN can only output values between -1 and 1 we need to scale the data to be between these values.

```python exec="on" source="material-block" session="dropout"
class MinMaxScaler:
    """A class which scales data to be within a chosen range"""

    def __init__(self, feature_range: tuple =(0, 1)):
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

scaler = MinMaxScaler(feature_range=(-1, 1))
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)
```

Now we need to construct our QNNs, they are comprised of feature maps and an ansatz. The ansatz contains CNOT gates with nearest neighbour entanglement after every rotation gate. The feature map contains two rotation gates RY and RZ, taking as rotation angles $\arcsin(x)$ and $\arccos(x^2)$ respectively.

```python exec="on" source="material-block" session="dropout"

def hea_ansatz(n_qubits, layer):
    """creates an ansatz which performs RX, RZ, RX rotations on each qubit,
    which nearest neighbour CNOT gates interpersed between each rotational gate."""
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

# the two feature maps we will be using
def fm1(n_qubits):
    return [pyq.RY(i, "x1") for i in range(n_qubits)]


def fm2(n_qubits):
    return [pyq.RZ(i, "x2") for i in range(n_qubits)]

# The class which constructs QNNs
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
        self.observable = pyq.Observable([pyq.Z(i) for i in range(n_qubits)]).to(
            device, dtype=torch.complex64
        )
        self.params = self.params.to(device=device, dtype=torch.float32)

    def build_operations(self):
        """defines operations for the quantum circuit and trainable parameters."""
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
        """constructs a QuantumCircuit object and puts it on device."""
        return pyq.QuantumCircuit(
            n_qubits=self.n_qubits,
            operations=operations,
        ).to(device=self.device, dtype=torch.complex64)

    def forward(self, x):
        """the forward pass for the QNN"""
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

# first we will define a QNN which is overparameterised with no dropout.
model = QuantumModelBase(n_qubits=n_qubits, n_layers=depth, device=device)
# define the corresponding optimizer for the problem
opt = optim.Adam(model.parameters(), lr=lr)
```

We now wish to train the QNN to learn the noisy sinusoidal function we have defined. This function will return the loss curves for the train and test sets.

```python exec="on" source="material-block" session="dropout"
def train(model, opt, x_train, y_train, x_test, y_test, epochs, device):
    # lists which will store losses for train and tests sets as we train.
    train_loss_history = []
    validation_loss_history = []

    x_test = tensor(x_test).to(device, dtype=torch.float32)
    y_test = (
        tensor(y_test)
        .to(device, dtype=torch.float32)
        .reshape(
            -1,
        )
    )

    x_train = tensor(x_train).to(device, dtype=torch.float32)
    y_train = (
        tensor(y_train)
        .to(device, dtype=torch.float32)
        .reshape(
            -1,
        )
    )

    # we will be using the mean squared error as our loss function.
    cost_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()

        opt.zero_grad()
        y_preds = model(x_train)
        train_loss = cost_fn(y_preds, y_train.flatten())
        train_loss.backward()
        opt.step()

        # no dropout is performed during evaluation of the model.
        model.eval()
        train_preds = model(x_train)
        train_loss = cost_fn(train_preds, y_train.flatten()).detach().cpu().numpy()

        test_preds = model(x_test)
        test_loss = cost_fn(test_preds, y_test.flatten()).detach().cpu().numpy()

        train_loss_history.append(train_loss)
        validation_loss_history.append(test_loss)

        # log performance every 100 epochs.
        if epoch % 100 == 0:
            print(f"epoch: {epoch}, train loss {train_loss}, val loss: {test_loss}")

    return train_loss_history, validation_loss_history

# train the vanilla QNN, extracting the training and testing loss curves
no_dropout_train_loss_hist, no_dropout_test_loss_hist = train(
    model=model,
    opt=opt,
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    epochs=epochs,
    device=device,
)
```

Next we define a QNN with the same archecture as before but now includes rotational dropout. Rotional dropout will randomly drop single qubit trainable parameterised gates with some probability dropout_prob.

```python exec="on" source="material-block" session="dropout"

class DropoutModel(QuantumModelBase):
    """Inherits from QuantumModelBase but the build_circuit function now creates a circuit which drops certain gates with some probability during training."""
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

# Define the QNN with rotational quantum dropout.
model = DropoutModel(
    n_qubits=n_qubits,
    n_layers=depth,
    device=device,
    dropout_mode=DropoutMode.ROTATIONAL,
    dropout_prob=dropout_prob,
)

# Define the corresponding optimiser
opt = optim.Adam(model.parameters(), lr=lr)

# train the QNN which contains rotational dropout.
dropout_train_loss_hist, dropout_test_loss_hist = train(
    model=model,
    opt=opt,
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    epochs=epochs,
    device=device,
)
```

Now that we have trained both a regular QNN and one with rotational quantum dropout, we can visualise the training and testing loss curves. What we observe is the regular QNN performing better on the train set but poorer on the test set. We can attribute this discrepency to the regular QNN overfitting to the noise rather than the true underlying function. The QNN with rotational dropout does no exhibit overfitting and adheres to the true function better, thus performs better on the test set.

```python exec="on" source="material-block" html="1" session="dropout"
def plot_results(
    no_dropout_train_history,
    no_dropout_test_history,
    dropout_train_history,
    dropout_test_history,
    epochs,
):
    fig, ax = plt.subplots(1, 2)
    plt.subplots_adjust(wspace=0.05)
    ax[0].set_title("MSE train")

    ax[0].plot(range(epochs), no_dropout_train_history, label="no dropout")
    ax[0].plot(range(epochs), dropout_train_history, label="dropout")

    ax[1].set_title("MSE test")
    ax[1].plot(range(epochs), no_dropout_test_history, label="no dropout")
    ax[1].plot(range(epochs), dropout_test_history, label="dropout")

    ax[0].legend(
        loc="upper center", bbox_to_anchor=(1.01, 1.25), ncol=4, fancybox=True, shadow=True
    )

    for ax in ax.flat:
        ax.set_xlabel("Epochs")
        ax.set_ylabel("MSE")
        ax.set_yscale("log")
        ax.set_ylim([1e-3, 0.6])
        ax.label_outer()

    plt.subplots_adjust(bottom=0.3)

    from io import StringIO  # markdown-exec: hide
    from matplotlib.figure import Figure  # markdown-exec: hide
    def fig_to_html(fig: Figure) -> str:  # markdown-exec: hide
        buffer = StringIO()  # markdown-exec: hide
        fig.savefig(buffer, format="svg")  # markdown-exec: hide
        return buffer.getvalue()  # markdown-exec: hide
    print(fig_to_html(plt.gcf())) # markdown-exec: hide


# finally we compare the
plot_results(
    no_dropout_train_history=no_dropout_train_loss_hist,
    dropout_train_history=dropout_train_loss_hist,
    no_dropout_test_history=no_dropout_test_loss_hist,
    dropout_test_history=dropout_test_loss_hist,
    epochs=epochs,
)
```
