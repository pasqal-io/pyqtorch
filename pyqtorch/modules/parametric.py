import torch
from torch.nn import Module
from numpy.typing import ArrayLike

from pyqtorch.core.utils import OPERATIONS_DICT
from pyqtorch.core.batched_operation import _apply_batch_gate
from pyqtorch.core.operation import create_controlled_matrix_from_operation, _apply_gate


class RotationGate(Module):
    def __init__(self, gate: str, qubits: ArrayLike, n_qubits: int, param_name: str):
        super().__init__()
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.param_name = param_name
        self.register_buffer("imat", OPERATIONS_DICT["I"])
        self.register_buffer("paulimat", OPERATIONS_DICT[gate])

    def forward(self, thetas: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        theta = thetas[self.param_name]
        batch_size = len(theta)
        mats = rot_matrices(theta, self.paulimat, self.imat, batch_size)
        return _apply_batch_gate(state, mats, self.qubits, self.n_qubits, batch_size)

    @property
    def device(self):
        return self.imat.device



def rot_matrices(
    theta: torch.Tensor, P: torch.Tensor, I: torch.Tensor, batch_size: int
) -> torch.Tensor:
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


def RX(*args, **kwargs):
    return RotationGate("X", *args, **kwargs)

def RY(*args, **kwargs):
    return RotationGate("Y", *args, **kwargs)

def RZ(*args, **kwargs):
    return RotationGate("Z", *args, **kwargs)
