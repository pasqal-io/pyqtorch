from __future__ import annotations

from typing import Any

import torch
from numpy.typing import ArrayLike
from torch.nn import Module

from pyqtorch.core.batched_operation import _apply_batch_gate
from pyqtorch.core.utils import OPERATIONS_DICT


class RotationGate(Module):
    def __init__(self, gate: str, qubits: ArrayLike, n_qubits: int, param_name: str):
        super().__init__()
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.param_name = param_name
        self.gate = gate
        self.register_buffer("imat", OPERATIONS_DICT["I"])
        self.register_buffer("paulimat", OPERATIONS_DICT[gate])

    def forward(self, thetas: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        theta = thetas[self.param_name]
        batch_size = len(theta)
        mats = rot_matrices(theta, self.paulimat, self.imat, batch_size)
        return _apply_batch_gate(state, mats, self.qubits, self.n_qubits, batch_size)

    @property
    def device(self) -> torch.device:
        return self.imat.device

    def extra_repr(self) -> str:
        return f"'{self.gate}', {self.qubits}, {self.n_qubits}, '{self.param_name}'"


def rot_matrices(
    theta: torch.Tensor, P: torch.Tensor, I: torch.Tensor, batch_size: int  # noqa: E741
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


def RX(*args: Any, **kwargs: Any) -> RotationGate:
    return RotationGate("X", *args, **kwargs)


def RY(*args: Any, **kwargs: Any) -> RotationGate:
    return RotationGate("Y", *args, **kwargs)


def RZ(*args: Any, **kwargs: Any) -> RotationGate:
    return RotationGate("Z", *args, **kwargs)
