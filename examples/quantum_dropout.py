import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import torch 
from torch import manual_seed, tensor, optim 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pyqtorch as pyq
from torch.autograd import grad
manual_seed(12345)
np.random.seed(12345)

def sin_dataset(dataset_size=100, test_size=0.4, noise=0.4):
    x_ax = np.linspace(-1,1,dataset_size)
    y = [[np.sin(x*np.pi)] for x in x_ax]
    np.random.seed(123)
    noise = np.array([np.random.normal(0,0.5,1) for i in y])*noise
    y = np.array(y+noise)
    x_train, x_test, y_train, y_test = train_test_split(x_ax, y, test_size=test_size, random_state=40, shuffle=True)
    x_train =  x_train.reshape(-1,1)
    x_test = x_test.reshape(-1,1)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    return x_train, x_test, y_train, y_test


def hea_ansatz(n_qubits, layer):
    ops = []
    for i in range(n_qubits):
        ops.append(pyq.RX(i, param_name=f"theta_{i}{layer}{0}"))
        ops.append(pyq.RZ(i, param_name=f"theta_{i}{layer}{1}"))
        ops.append(pyq.RX(i, param_name=f"theta_{i}{layer}{2}"))

    for j in range(n_qubits-1):
        ops.append(pyq.CNOT(control=j, target=j+1))
    return pyq.QuantumCircuit(n_qubits, ops)


def fm1(n_qubits):
    ops = []
    for i in range(n_qubits):
        ops.append(pyq.RY(i))
    return pyq.QuantumCircuit(n_qubits, ops)


def fm2(n_qubits):
    ops = []
    for i in range(n_qubits):
        ops.append(pyq.RZ(i))
    return pyq.QuantumCircuit(n_qubits, ops)


class Model(torch.nn.Module):

    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits 
        self.n_layers = n_layers
        self.ansatz_collection = [hea_ansatz(n_qubits, i) for i in range(n_layers)]
        param_tensor = torch.nn.Parameter(torch.rand((n_qubits*3*n_layers), requires_grad=True))
        self.params = self.convert_tensor_to_dict(param_tensor)
        self.embedding1 = fm1(n_qubits=n_qubits)
        self.embedding2 = fm2(n_qubits=n_qubits)
        self.observable = pyq.Z(0)

    def convert_tensor_to_dict(self, params):
        param_dict = {}
        count = 0
        for i in range(self.n_qubits):
            for j in range(self.n_layers):
                for k in range(3):
                    param_dict[f"theta_{i}{j}{k}"] = params[count]
                    count +=1
        return torch.nn.ParameterDict(param_dict)

    def forward(self, x):
        batch_size = len(x)
        x_1 = np.arcsin(x)
        x_2 = np.arccos(x**2)
        state = pyq.zero_state(n_qubits=self.n_qubits, batch_size=batch_size)
        for i in range(self.n_layers):
            state = self.embedding1(state, x_1)
            state = self.embedding2(state, x_2)
            state = self.ansatz_collection[i](state, self.params)
        
        new_state = self.observable(state).reshape((2**n_qubits, batch_size))
        state = state.reshape((2**self.n_qubits, batch_size))

        return torch.real(torch.sum(torch.conj(state)*new_state, axis=0))


def train_step(model, opt, data):
    opt.zero_grad()
    y_true = data[1]
    y_preds = model(data[0])
    loss = torch.nn.MSELoss()(y_preds, y_true)
    loss.backward()
    opt.step()

    return loss


def train(model, opt, x_train, y_train, x_test, y_test, epochs, batch_size):
    train_loss_history = []
    validation_loss_history = []
    steps_per_epoch = x_train.shape[0] // batch_size

    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            x_batch = tensor(x_train[step*batch_size:(step+1)*batch_size])
            y_batch = tensor(y_train[step*batch_size:(step+1)*batch_size])

            loss = train_step(model, opt, (x_batch, y_batch))

        test_preds = model(tensor(x_test))
        test_loss = torch.nn.MSELoss()(test_preds, tensor(y_test))
        train_loss_history.append(loss.detach().numpy())
        validation_loss_history.append(test_loss.detach().numpy())

        if epoch % 100 == 0:
            print(f"epoch: {epoch}, train loss {loss.detach().numpy()}, val loss: {test_loss.detach().numpy()}")

    plt.plot(train_loss_history,label="train loss")
    plt.plot(validation_loss_history,label="val loss")
    plt.xlabel(r"epochs")
    plt.ylabel(r"mse loss")
    plt.legend()
    plt.show()

    return model, train_loss_history, validation_loss_history

if __name__ == "__main__":
    n_qubits = 4
    depth = 10
    lr = 0.01
    n_points = 20
    epochs = 700
    
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
    plt.show()

    scaler = MinMaxScaler(feature_range=(-1,1))
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.transform(y_test)
    

    """
    Currently this is my small working example for a feature map taking in data
    of the form (batch_size, features). However, I encounter the error:

    line 70, in _unitary
    cos_t = cos_t.repeat((2, 2, 1))
            ^^^^^^^^^^^^^^^^^^^^^^^
    RuntimeError: Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor
    """
    def fm1(n_qubits):
        ops = []
        for i in range(n_qubits):
            ops.append(pyq.RY(i))
        return pyq.QuantumCircuit(n_qubits, ops)

    n_qubits = 4
    batch_size = 2
    state = pyq.zero_state(n_qubits=n_qubits, batch_size=batch_size)
    data = torch.rand(batch_size, n_qubits)
    fm = fm1(n_qubits)
    fm_out = fm(state, data)
    print(fm_out)

    #print(v.shape)
    #fm_out = fm(pyq.zero_state(n_qubits=4, batch_size=1), v)
    #hea = hea_ansatz(4, 1)
    #params = {"theta_010": torch.rand(1), "theta_011": torch.rand(1), "theta_012":torch.rand(1), 
    #          "theta_110": torch.rand(1), "theta_111": torch.rand(1), "theta_112": torch.rand(1),
    #          "theta_210": torch.rand(1), "theta_211": torch.rand(1), "theta_212":torch.rand(1),
    #          "theta_310": torch.rand(1), "theta_311": torch.rand(1), "theta_312":torch.rand(1)}
    #hea_out = hea(pyq.zero_state(4), params)
    #print("TRAINING")
    #model = Model(n_qubits=n_qubits, n_layers=depth)
    #print(model)
    #print(model(torch.rand(1, 4)))
    #opt = optim.Adam(model.parameters(), lr=lr)
    #_, train_loss_hist, val_loss_hist = train(model=model, opt=opt, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, epochs=epochs, batch_size=15)
