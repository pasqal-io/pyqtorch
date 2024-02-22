from __future__ import annotations

import torch

import pyqtorch as pyq
from pyqtorch.parametric import Parametric

N_DEVICES = 2

assert torch.cuda.device_count() == N_DEVICES


class ParallelCircuit(torch.nn.Module):
    def __init__(self, c0: pyq.QuantumCircuit, c1: pyq.QuantumCircuit, params_c0, params_c1):
        super().__init__()
        self.c0 = c0.to("cuda:0")
        self.c1 = c1.to("cuda:1")
        self.params_c0 = torch.nn.ParameterDict({k: v.to("cuda:0") for k, v in params_c0.items()})
        self.params_c1 = torch.nn.ParameterDict({k: v.to("cuda:1") for k, v in params_c1.items()})

    def forward(self, state: torch.Tensor, values: dict = dict()) -> torch.Tensor:
        state = self.c0.forward(state.to("cuda:0"), self.params_c0)
        return self.c1.forward(state.to("cuda:1"), self.params_c1)


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


params_c0 = init_params(c0)
params_c1 = init_params(c1)

circ = ParallelCircuit(c0, c1, params_c0, params_c1)
new_state = circ.forward(pyq.zero_state(n_qubits))
