from cgi import parse_multipart
import torch
from pyqtorch.core.operation import X, Y, Z

qubit_operators = {'X': X, 'Y': Y, 'Z': Z}


def total_magnetization(state, N_qubits, batch_size):
    new_state = torch.zeros_like(state)

    for i in range(N_qubits):
        new_state += Z(state, [i], N_qubits)

    state = state.reshape((2**N_qubits, batch_size))
    new_state = new_state.reshape((2**N_qubits, batch_size))

    return torch.real(
        torch.sum(torch.conj(state) * new_state, axis=0))


def measure_openfermion(state, operator, N_qubits, batch_size):
    new_state = torch.zeros_like(state)

    for op, coef in operator.terms.items():
        for qubit, pauli in op:
            state_bis = qubit_operators[pauli](state, [qubit], N_qubits)
            new_state += state_bis * coef

    state = state.reshape((2**N_qubits, batch_size))
    new_state = new_state.reshape((2**N_qubits, batch_size))

    return torch.real(
        torch.sum(torch.conj(state) * new_state, axis=0))
