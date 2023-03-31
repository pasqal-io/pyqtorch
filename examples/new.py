import torch
from torch import Tensor
from torch.nn import Module
from numpy.typing import ArrayLike

from pyqtorch.core.utils import OPERATIONS_DICT
from pyqtorch.core.batched_operation import _apply_batch_gate

    # theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int
    # dev = state.device
    # batch_size = len(theta)
    # mat = get_parametrized_batch_for_operation("X", theta, batch_size, dev)
    # _apply_batch_gate(state, mat, qubits, N_qubits, batch_size)


class RotationGate(Module):
    def __init__(
        self, gate: str, qubits: ArrayLike, n_qubits: int
    ):
        super().__init__()
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.register_buffer("I", OPERATIONS_DICT["I"])
        self.register_buffer("P", OPERATIONS_DICT[gate])

    def forward(self, theta: Tensor, state: Tensor) -> Tensor:
        batch_size = len(theta)
        mats = rot_matrices(theta, self.P, self.I, batch_size)
        return _apply_batch_gate(state, mats, self.qubits, self.n_qubits, batch_size)


def rot_matrices(theta: Tensor, P: Tensor, I: Tensor, batch_size: int) -> Tensor:
    """ 
    Returns:
        torch.Tensor: a batch of gates after applying theta
    """
    cos_t = torch.cos(theta / 2).unsqueeze(0).unsqueeze(1)
    cos_t = cos_t.repeat((2, 2, 1))
    sin_t = torch.sin(theta / 2).unsqueeze(0).unsqueeze(1)
    sin_t = sin_t.repeat((2, 2, 1))

    batch_imat = I.unsqueeze(2).repeat(1, 1, batch_size)
    batch_operation_mat = P.unsqueeze(2).repeat(1, 1, batch_size)

    return cos_t * batch_imat - 1j * sin_t * batch_operation_mat


def RX(qubits, n_qubits, **kwargs):
    return RotationGate("X", qubits, n_qubits, **kwargs)

if __name__ == "__main__":
    from pyqtorch.core.batched_operation import batchedRX
    import time

    #print(OPERATIONS_DICT["I"])
    #g = OPERATIONS_DICT["I"].to(device="mps", dtype=torch.cfloat)
    #print(g + g)

    dtype = torch.cfloat
    device = "cpu"

    theta = torch.ones(10000, device=device) * 3.14
    state_00 = torch.tensor([[1, 0], [0, 0]], dtype=dtype, device=device).unsqueeze(2)
    state_10 = torch.tensor([[0, 1], [0, 0]], dtype=dtype, device=device).unsqueeze(2)
    state_01 = torch.tensor([[0, 0], [1, 0]], dtype=dtype, device=device).unsqueeze(2)
    state_11 = torch.tensor([[0, 0], [0, 1]], dtype=dtype, device=device).unsqueeze(2)

    qubits = [0]
    n_qubits = 2

    #t1 = time.time()
    #s = batchedRX(theta, state_00, qubits, n_qubits)
    #print(f"{time.time()-t1}")
    #print(s)

    gate = RX(qubits, n_qubits).to(device=device, dtype=dtype)
    gate = torch.compile(gate)
    t1 = time.time()
    s = gate(theta, state_00)
    print(f"{time.time()-t1}")
    print(s)
