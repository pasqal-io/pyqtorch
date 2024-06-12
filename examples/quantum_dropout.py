from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import manual_seed, optim, tensor

import pyqtorch as pyq
from pyqtorch.parametric import Parametric
from pyqtorch.circuit import DropoutQuantumCircuit

manual_seed(12345)
np.random.seed(12345)


def sin_dataset(dataset_size=100, test_size=0.4, noise=0.4):
    x_ax = np.linspace(-1, 1, dataset_size)
    y = [[np.sin(x * np.pi)] for x in x_ax]
    np.random.seed(123)
    noise = np.array([np.random.normal(0, 0.5, 1) for i in y]) * noise
    y = np.array(y + noise)
    x_train, x_test, y_train, y_test = train_test_split(
        x_ax, y, test_size=test_size, random_state=40, shuffle=True
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
        ops.append(pyq.RZ(i, param_name=f"theta_{i}{layer}{1}"))
        ops.append(pyq.RX(i, param_name=f"theta_{i}{layer}{2}"))

    for j in range(n_qubits - 1):
        ops.append(pyq.CNOT(control=j, target=j + 1))
    return ops


def fm1(n_qubits):
    ops = []
    for i in range(n_qubits):
        ops.append(pyq.RY(i, "x1"))
    return ops


def fm2(n_qubits):
    ops = []
    for i in range(n_qubits):
        ops.append(pyq.RZ(i, "x2"))
    return ops


class DropoutModel(torch.nn.Module):
    def __init__(self, n_qubits, n_layers, device, dropout_mode="rotational_dropout", dropout_prob=0.1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.embedding1 = fm1(n_qubits=n_qubits)
        self.embedding2 = fm2(n_qubits=n_qubits)

        params = {}
        operations = []
        for i in range(n_layers):
            operations += self.embedding1 + self.embedding2
            layer_i_ansatz = hea_ansatz(n_qubits=n_qubits, layer=i)
            operations += layer_i_ansatz
            for op in layer_i_ansatz:
                if isinstance(op, Parametric):
                    params[f"{op.param_name}"] = torch.randn(1, requires_grad=True)

        params = torch.nn.ParameterDict(params)
        
        circuit = DropoutQuantumCircuit(
            n_qubits=n_qubits,
            operations=operations,
            dropout_mode=dropout_mode,
            dropout_prob=dropout_prob,
        )

        observable = pyq.QuantumCircuit(n_qubits, [pyq.Z(i) for i in range(n_qubits)])
        self.circuit = circuit.to(device=device, dtype=torch.complex64)
        self.observable = observable.to(device=device, dtype=torch.complex64)
        self.params = params.to(device=device, dtype=torch.float32)

    def forward(self, x):
        x = x.flatten()
        x_1 = {"x1": torch.asin(x)}
        x_2 = {"x2": torch.acos(x**2)}
        
        self.circuit.training = self.training
        
        state = self.circuit.init_state(batch_size=int(x.shape[0]))
        out = pyq.expectation(
            circuit=self.circuit,
            state=state,
            values={**self.params, **x_1, **x_2},
            observable=self.observable,
        )

        return out


class RegularModel(torch.nn.Module):
    def __init__(self, n_qubits, n_layers, device):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.embedding1 = fm1(n_qubits=n_qubits)
        self.embedding2 = fm2(n_qubits=n_qubits)

        params = {}
        operations = []
        for i in range(n_layers):
            operations += self.embedding1 + self.embedding2
            layer_i_ansatz = hea_ansatz(n_qubits=n_qubits, layer=i)
            operations += layer_i_ansatz
            for op in layer_i_ansatz:
                if isinstance(op, Parametric):
                    params[f"{op.param_name}"] = torch.randn(1, requires_grad=True)

        params = torch.nn.ParameterDict(params)
        circuit = pyq.QuantumCircuit(
            n_qubits=n_qubits,
            operations=operations,
        )

        observable = pyq.QuantumCircuit(n_qubits, [pyq.Z(i) for i in range(n_qubits)])
        self.circuit = circuit.to(device=device, dtype=torch.complex64)
        self.observable = observable.to(device=device, dtype=torch.complex64)
        self.params = params.to(device=device, dtype=torch.float32)

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



def train_step(model, opt, data):
    opt.zero_grad()
    y_true = data[1].flatten()
    y_preds = model(data[0])
    loss = torch.nn.MSELoss()(y_preds, y_true)
    loss.backward()
    opt.step()

    return loss


def train(model, opt, x_train, y_train, x_test, y_test, epochs, batch_size):
    train_loss_history = []
    validation_loss_history = []
    steps_per_epoch = x_train.shape[0] // batch_size

    x_test = tensor(x_test).to(device, dtype=torch.float32)
    y_test = tensor(y_test).to(device, dtype=torch.float32).flatten()

    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            model.train()
            x_batch = tensor(x_train[step * batch_size : (step + 1) * batch_size]).to(
                device, dtype=torch.float32
            )
            y_batch = tensor(y_train[step * batch_size : (step + 1) * batch_size]).to(
                device, dtype=torch.float32
            )

            loss = train_step(model, opt, (x_batch, y_batch))

        model.eval()
        test_preds = model(x_test)
        test_loss = torch.nn.MSELoss()(test_preds, y_test).detach().numpy()

        train_preds = model(x_batch)
        train_loss = torch.nn.MSELoss()(train_preds, y_batch.flatten()).detach().numpy()

        train_loss_history.append(train_loss)
        validation_loss_history.append(test_loss)

        if epoch % 100 == 0:
            print(f"epoch: {epoch}, train loss {train_loss}, val loss: {test_loss}")

    plt.plot(train_loss_history, label="train loss")
    plt.plot(validation_loss_history, label="val loss")
    plt.xlabel(r"epochs")
    plt.ylabel(r"mse loss")
    plt.legend()
    plt.savefig("loss_curve.pdf")
    plt.close()
    return model, train_loss_history, validation_loss_history


if __name__ == "__main__":
    device = torch.device("cpu")
    n_qubits = 4
    depth = 10
    lr = 0.01
    n_points = 20
    epochs = 1000

    x_train, x_test, y_train, y_test = sin_dataset(dataset_size=n_points, test_size=0.25)
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
    plt.savefig("function.pdf")
    plt.close()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.transform(y_test)

    print("TRAINING")
    model = RegularModel(n_qubits=n_qubits, n_layers=depth, device=device)
    no_drop_p = model.params
    opt = optim.Adam(model.parameters(), lr=lr)
    _, train_loss_hist, val_loss_hist = train(
        model=model,
        opt=opt,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=epochs,
        batch_size=15,
    )

    print("DROPOUT TRAINING")
    model = DropoutModel(
        n_qubits=n_qubits,
        n_layers=depth,
        dropout_mode="rotational_dropout",
        dropout_prob=0.03,
        device=device,
    )
    model.params = no_drop_p
    drop_p = model.params
    print("DO PARAMS MATCH: ", no_drop_p == drop_p)
    opt = optim.Adam(model.parameters(), lr=lr)
    _, train_loss_hist, val_loss_hist = train(
        model=model,
        opt=opt,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=epochs,
        batch_size=15,
    )