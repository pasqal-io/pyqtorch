from __future__ import annotations

import timeit
from functools import reduce
from operator import add

import numpy as np
import torch
from torch.nn.functional import mse_loss

import pyqtorch as pyq
from pyqtorch.parametric import Parametric

N_DEVICES = 2
N_QUBITS = 2
N_POINTS = 100
N_EPOCHS = 1


def fn(x: torch.Tensor, degree: int) -> torch.Tensor:
    return 0.05 * reduce(add, (torch.cos(i * x) + torch.sin(i * x) for i in range(degree)), 0)


x = torch.linspace(0, 10, N_POINTS)
y = fn(x, 5)

assert torch.cuda.is_available()
assert torch.cuda.device_count() == N_DEVICES


def init_params(circ: pyq.QuantumCircuit, device: torch.device) -> torch.nn.ParameterDict:
    return torch.nn.ParameterDict(
        {
            op.param_name: torch.rand(1, requires_grad=True, device=device)
            for op in circ.operations
            if isinstance(op, Parametric)
        }
    )


def hea(n_qubits: int, n_layers: int, param_name: str) -> list:
    ops = []
    for layer in range(n_layers):
        ops += [pyq.RX(i, f"{param_name}_0_{layer}_{i}") for i in range(n_qubits)]
        ops += [pyq.RY(i, f"{param_name}_1_{layer}_{i}") for i in range(n_qubits)]
        ops += [pyq.RX(i, f"{param_name}_2_{layer}_{i}") for i in range(n_qubits)]
        ops += [pyq.CNOT(i % n_qubits, (i + 1) % n_qubits) for i in range(n_qubits)]
    return ops


class SingleDeviceCircuit(torch.nn.Module):
    def __init__(self, n_qubits: int = N_QUBITS):
        super().__init__()
        self.feature_map = pyq.QuantumCircuit(
            n_qubits, [pyq.RX(i, "x") for i in range(n_qubits)]
        ).to("cuda:1")
        self.c0 = pyq.QuantumCircuit(n_qubits, hea(n_qubits, 1, "theta")).to("cuda:1")
        self.c1 = pyq.QuantumCircuit(n_qubits, hea(n_qubits, 1, "phi")).to("cuda:1")
        self.params_c0 = init_params(self.c0, device="cuda:1")
        self.params_c1 = init_params(self.c1, device="cuda:1")
        self.observable = pyq.Z(0).to("cuda:1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = pyq.zero_state(N_QUBITS).to("cuda:1")
        state = self.feature_map.forward(state.to("cuda:1"), {"x": x.to("cuda:1")})
        state = self.c0.forward(state, self.params_c0)
        state = self.c1.forward(state.to("cuda:1"), self.params_c1)
        projected = self.observable.forward(state)
        return pyq.inner_prod(state, projected).real


class ModelParallelCircuit(torch.nn.Module):
    def __init__(self, n_qubits: int = N_QUBITS):
        super().__init__()
        self.feature_map = pyq.QuantumCircuit(
            n_qubits, [pyq.RX(i, "x") for i in range(n_qubits)]
        ).to("cuda:0")
        self.c0 = pyq.QuantumCircuit(n_qubits, hea(n_qubits, 1, "theta")).to("cuda:0")
        self.c1 = pyq.QuantumCircuit(n_qubits, hea(n_qubits, 1, "phi")).to("cuda:1")
        self.params_c0 = init_params(self.c0, device="cuda:0")
        self.params_c1 = init_params(self.c1, device="cuda:1")
        self.observable = pyq.Z(0).to("cuda:1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = pyq.zero_state(N_QUBITS).to("cuda:0")
        state = self.feature_map.forward(state.to("cuda:0"), {"x": x.to("cuda:0")})
        state = self.c0.forward(state, self.params_c0)
        state = self.c1.forward(state.to("cuda:1"), self.params_c1)
        projected = self.observable.forward(state)
        return pyq.inner_prod(state, projected).real


def train(circ) -> None:
    optimizer = torch.optim.Adam(
        {**circ.params_c0, **circ.params_c1}.values(), lr=0.01, foreach=False
    )
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        y_pred = circ.forward(x)
        loss = mse_loss(y_pred, y.to("cuda:1"))
        loss.backward()
        optimizer.step()


class PipelineParallelCircuit(ModelParallelCircuit):
    def __init__(self, split_size=N_POINTS // 2, *args, **kwargs):
        super(PipelineParallelCircuit, self).__init__(*args, **kwargs)
        self.split_size = split_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        init_state = pyq.zero_state(N_QUBITS).to("cuda:0")
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.feature_map(init_state, {"x": s_next.to("cuda:0")})
        s_prev = self.c0.forward(s_prev, self.params_c0).to("cuda:1")
        ret = []

        for s_next in splits:
            s_prev = self.c1.forward(s_prev, self.params_c1)
            ret.append(pyq.inner_prod(s_prev, self.observable.forward(s_prev)).real)

            s_prev = self.feature_map(init_state, {"x": s_next.to("cuda:0")})
            s_prev = self.c0.forward(s_prev, self.params_c0).to("cuda:1")

        s_prev = self.c1.forward(s_prev, self.params_c1)
        ret.append(pyq.inner_prod(s_prev, self.observable.forward(s_prev)).real)
        return torch.cat(ret)


if __name__ == "__main__":
    res = {}
    for model_cls in [SingleDeviceCircuit, ModelParallelCircuit, PipelineParallelCircuit]:
        setup = "circ = model_cls(N_QUBITS)"
        pp_run_times = timeit.repeat("train(circ)", setup, number=1, repeat=10, globals=globals())
        pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)
        res[model_cls.__name__] = pp_mean
    print(res)
