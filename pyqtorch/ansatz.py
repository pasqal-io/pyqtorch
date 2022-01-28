import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

from pyqtorch.core.circuit import QuantumCircuit
from pyqtorch.core.operation import RX, RY, RZ, U, CNOT


class OneLayerRotation(QuantumCircuit):

    def __init__(self, n_qubits, arbitrary=False):
        super().__init__(n_qubits)
        if arbitrary:
            self.theta = nn.Parameter(torch.empty((self.n_qubits, 3)))
        else:
            self.theta = nn.Parameter(torch.empty((self.n_qubits, )))
        self.reset_parameters()
        self.arbitrary=arbitrary

    def reset_parameters(self):
        init.uniform_(self.theta, -2 * np.pi, 2 * np.pi)

    def forward(self, state):
        if self.arbitrary:
            for i, t in enumerate(self.theta):
                state = U(t[0], t[1], t[2], state, [i], self.n_qubits)
        return state


class OneLayerXRotation(OneLayerRotation):

    def __init__(self, n_qubits):
        super().__init__(n_qubits)

    def forward(self, state):
        for i, t in enumerate(self.theta):
            state = RX(t, state, [i], self.n_qubits)
        return state


class OneLayerYRotation(OneLayerRotation):

    def __init__(self, n_qubits):
        super().__init__(n_qubits)

    def forward(self, state):
        for i, t in enumerate(self.theta):
            state = RY(t, state, [i], self.n_qubits)
        return state


class OneLayerZRotation(OneLayerRotation):

    def __init__(self, n_qubits):
        super().__init__(n_qubits)

    def forward(self, state):
        for i, t in enumerate(self.theta):
            state = RZ(t, state, [i], self.n_qubits)
        return state


class OneLayerEntanglingAnsatz(QuantumCircuit):

    def __init__(self, n_qubits):
        super().__init__(n_qubits)
        self.param_layer = OneLayerRotation(n_qubits=self.n_qubits, arbitrary=True)

    def forward(self, state):
        state = self.param_layer(state)
        for i in range(self.n_qubits):
            state = CNOT(state, [i % self.n_qubits, (i+1) % self.n_qubits],
                                self.n_qubits)
        return state


class AlternateLayerAnsatz(QuantumCircuit):

    def __init__(self, n_qubits, n_layers):
        super().__init__(n_qubits)
        self.layers = nn.ModuleList([OneLayerEntanglingAnsatz(self.n_qubits)
                        for _ in range(n_layers)])

    def forward(self, state):
        for i, layer in enumerate(self.layers):
            state = layer(state)
        return state
