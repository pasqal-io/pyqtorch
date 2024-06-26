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
