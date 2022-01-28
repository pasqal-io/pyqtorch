import torch
import torch.nn as nn

class QuantumCircuit(nn.Module):

    def __init__(self, n_qubits):
        super(QuantumCircuit, self).__init__()
        self.n_qubits = n_qubits

    def init_state(self, batch_size=1, device='cpu'):
        state = torch.zeros((2**self.n_qubits, batch_size),
                            dtype=torch.cdouble).to(device)
        state[0] = 1
        state = state.reshape([2] * self.n_qubits + [batch_size])
        return state

    def uniform_state(self, batch_size=1, device='cpu'):
        state = torch.ones((2**self.n_qubits, batch_size),
                            dtype=torch.cdouble).to(device)
        state = state / torch.sqrt(torch.tensor(2**self.n_qubits))
        state = state.reshape([2] * self.n_qubits + [batch_size])
        return state
