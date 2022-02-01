import torch
from pyqtorch.core.utils import _apply_gate, _apply_batch_gate


IMAT = torch.eye(2, dtype=torch.cdouble)
XMAT = torch.tensor([[0, 1], [1, 0]], dtype=torch.cdouble)
YMAT = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cdouble)
ZMAT = torch.tensor([[1, 0], [0, -1]], dtype=torch.cdouble)


def RX(theta, state, qubits, N_qubits):
    dev = state.device
    mat = IMAT.to(dev) * torch.cos(theta/2) -\
        1j * XMAT.to(dev) * torch.sin(theta/2)
    return _apply_gate(state, mat, qubits, N_qubits)


def RY(theta, state, qubits, N_qubits):
    dev = state.device
    mat = IMAT.to(dev) * torch.cos(theta/2) -\
        1j * YMAT.to(dev) * torch.sin(theta/2)
    return _apply_gate(state, mat, qubits, N_qubits)


def RZ(theta, state, qubits, N_qubits):
    dev = state.device
    mat = IMAT.to(dev) * torch.cos(theta/2) +\
        1j * ZMAT.to(dev) * torch.sin(theta/2)
    return _apply_gate(state, mat, qubits, N_qubits)


def U(phi, theta, omega, state, qubits, N_qubits):
    '''U(phi, theta, omega) = RZ(omega)RY(theta)RZ(phi)'''
    dev = state.device
    t_plus = torch.exp(-1j * (phi + omega) / 2)
    t_minus = torch.exp(-1j * (phi - omega) / 2)
    mat = torch.tensor([[1, 0], [0, 0]], dtype=torch.cdouble).to(dev) \
        * torch.cos(theta/2) * t_plus -\
        torch.tensor([[0, 1], [0, 0]], dtype=torch.cdouble).to(dev) \
        * torch.sin(theta/2) * torch.conj(t_minus) +\
        torch.tensor([[0, 0], [1, 0]], dtype=torch.cdouble).to(dev) \
        * torch.sin(theta/2) * t_minus + \
        torch.tensor([[0, 0], [0, 1]], dtype=torch.cdouble).to(dev) \
        * torch.cos(theta/2) * torch.conj(t_plus)
    return _apply_gate(state, mat, qubits, N_qubits)


def X(state, qubits, N_qubits):
    dev = state.device
    mat = XMAT.to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def Z(state, qubits, N_qubits):
    dev = state.device
    mat = ZMAT.to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def Y(state, qubits, N_qubits):
    dev = state.device
    mat = YMAT.to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def H(state, qubits, N_qubits):
    dev = state.device
    mat = 1 / torch.sqrt(torch.tensor(2)) * \
        torch.tensor([[1, 1], [1, -1]], dtype=torch.cdouble).to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def CNOT(state, qubits, N_qubits):
    dev = state.device
    mat = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0],
                        [0, 0, 0, 1], [0, 0, 1, 0]],
                        dtype=torch.cdouble).to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def batchedRX(theta, state, qubits, N_qubits):
    dev = state.device
    batch_size = len(theta)

    cos_t = torch.cos(theta/2).unsqueeze(0).unsqueeze(1)
    cos_t = cos_t.repeat((2, 2, 1))
    sin_t = torch.sin(theta/2).unsqueeze(0).unsqueeze(1)
    sin_t = sin_t.repeat((2, 2, 1))

    imat = IMAT.unsqueeze(2).repeat(1, 1, batch_size).to(dev)
    xmat = XMAT.unsqueeze(2).repeat(1, 1, batch_size).to(dev)

    mat = cos_t * imat - 1j * sin_t * xmat

    return _apply_batch_gate(state, mat, qubits, N_qubits, batch_size)

def batchedRY(theta, state, qubits, N_qubits):
    dev = state.device
    batch_size = len(theta)

    cos_t = torch.cos(theta/2).unsqueeze(0).unsqueeze(1)
    cos_t = cos_t.repeat((2, 2, 1))
    sin_t = torch.sin(theta/2).unsqueeze(0).unsqueeze(1)
    sin_t = sin_t.repeat((2, 2, 1))

    imat = IMAT.unsqueeze(2).repeat(1, 1, batch_size).to(dev)
    xmat = YMAT.unsqueeze(2).repeat(1, 1, batch_size).to(dev)

    mat = cos_t * imat - 1j * sin_t * xmat

    return _apply_batch_gate(state, mat, qubits, N_qubits, batch_size)
