import torch
from pyqtorch.core.operation import Z


def total_magnetization(state, N_qubits, batch_size):
    new_state = torch.zeros_like(state)

    for i in range(N_qubits):
        new_state += Z(state, [i], N_qubits)

    state = state.reshape((2**N_qubits, batch_size))
    new_state = new_state.reshape((2**N_qubits, batch_size))

    return torch.real(
        torch.sum(torch.conj(state) * new_state, axis=0))
