from __future__ import annotations

from functools import reduce
from operator import add

import torch
from torch.nn.functional import mse_loss

import pyqtorch as pyq
from pyqtorch.parametric import Parametric

N_DEVICES = 2

assert torch.cuda.is_available()
assert torch.cuda.device_count() == N_DEVICES


class ParallelCircuit(torch.nn.Module):
    def __init__(self, c0: pyq.QuantumCircuit, c1: pyq.QuantumCircuit, params_c0, params_c1):
        super().__init__()
        self.feature_map = pyq.QuantumCircuit(
            n_qubits, [pyq.RX(i, "x") for i in range(n_qubits)]
        ).to("cuda:0")
        self.c0 = c0.to("cuda:0")
        self.c1 = c1.to("cuda:1")
        self.params_c0 = torch.nn.ParameterDict({k: v.to("cuda:0") for k, v in params_c0.items()})
        self.params_c1 = torch.nn.ParameterDict({k: v.to("cuda:1") for k, v in params_c1.items()})
        self.observable = pyq.Z(0).to("cuda:1")

    def forward(self, state: torch.Tensor, inputs: dict = dict()) -> torch.Tensor:
        state = self.feature_map.forward(
            state.to("cuda:0"), {k: v.to("cuda:0") for k, v in inputs.items()}
        )
        state = self.c0.forward(state, self.params_c0)
        state = self.c1.forward(state.to("cuda:1"), self.params_c1)
        projected = self.observable.forward(state)
        return pyq.inner_prod(state, projected).real


def hea(n_qubits: int, n_layers: int, param_name: str) -> list:
    ops = []
    for layer in range(n_layers):
        ops += [pyq.RX(i, f"{param_name}_0_{layer}_{i}") for i in range(n_qubits)]
        ops += [pyq.RY(i, f"{param_name}_1_{layer}_{i}") for i in range(n_qubits)]
        ops += [pyq.RX(i, f"{param_name}_2_{layer}_{i}") for i in range(n_qubits)]
        ops += [pyq.CNOT(i % n_qubits, (i + 1) % n_qubits) for i in range(n_qubits)]
    return ops


n_qubits = 2
c0 = pyq.QuantumCircuit(n_qubits, hea(n_qubits, 1, "theta"))
c1 = pyq.QuantumCircuit(n_qubits, hea(n_qubits, 1, "phi"))


def init_params(circ: pyq.QuantumCircuit) -> torch.nn.ParameterDict:
    return torch.nn.ParameterDict(
        {
            op.param_name: torch.rand(1, requires_grad=True)
            for op in circ.operations
            if isinstance(op, Parametric)
        }
    )


# Target function and some training data
def fn(x, degree):
    return 0.05 * reduce(add, (torch.cos(i * x) + torch.sin(i * x) for i in range(degree)), 0)


x = torch.linspace(0, 10, 100)
y = fn(x, 5)


params_c0 = init_params(c0)
params_c1 = init_params(c1)

circ = ParallelCircuit(c0, c1, params_c0, params_c1)
state = pyq.zero_state(n_qubits)


def exp_fn(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return pyq.expectation(
        circ, state, {**circ.params_c0, **circ.params_c1, **inputs}, circ.observable, "ad"
    )


optimizer = torch.optim.Adam({**circ.params_c0, **circ.params_c1}.values(), lr=0.01, foreach=False)
epochs = 10

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = circ.forward(pyq.zero_state(n_qubits), {"x": x})
    loss = mse_loss(y_pred, y.to("cuda:1"))
    loss.backward()
    print(f"{epoch}:{loss.item()}")
    optimizer.step()
