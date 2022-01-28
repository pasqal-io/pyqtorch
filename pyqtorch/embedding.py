from pyqtorch.core.circuit import QuantumCircuit
from pyqtorch.core.operation import batchedRX


class SingleLayerEncoding(QuantumCircuit):

    def __init__(self, n_qubits):
        super().__init__(n_qubits)

    def forward(self, state, x):
        for i in range(self.n_qubits):
            state = batchedRX(x, state, [i], self.n_qubits)
        return state
