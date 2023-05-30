from __future__ import annotations

import torch
from numpy.typing import ArrayLike
from torch.nn import Module

from pyqtorch.core.batched_operation import (
    _apply_batch_gate,
    create_controlled_batch_from_operation,
)
from pyqtorch.core.utils import OPERATIONS_DICT


class RotationGate(Module):
    n_params = 1

    def __init__(self, gate: str, qubits: ArrayLike, n_qubits: int):
        super().__init__()
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.gate = gate
        self.register_buffer("imat", OPERATIONS_DICT["I"])
        self.register_buffer("paulimat", OPERATIONS_DICT[gate])

    def matrices(self, thetas: torch.Tensor) -> torch.Tensor:
        # NOTE: thetas are assumed to be of shape (1,batch_size) or (batch_size,) because we
        # want to allow e.g. (3,batch_size) in the U gate.
        theta = thetas.squeeze(0) if thetas.ndim == 2 else thetas
        batch_size = len(theta)
        return rot_matrices(theta, self.paulimat, self.imat, batch_size)

    def apply(self, matrices: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = matrices.size(-1)
        return _apply_batch_gate(state, matrices, self.qubits, self.n_qubits, batch_size)

    def forward(self, state: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        mats = self.matrices(thetas)
        return self.apply(mats, state)

    def extra_repr(self) -> str:
        return f"qubits={self.qubits}, n_qubits={self.n_qubits}"


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


class U(Module):
    """Parametrized arbitrary rotation along the axes of the Bloch sphere

    The angles `phi, theta, omega` in tensor format, applied as:

        U(phi, theta, omega) = RZ(omega)RY(theta)RZ(phi)
    """

    n_params = 3

    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__()
        self.qubits = qubits
        self.n_qubits = n_qubits

        self.register_buffer("a", torch.tensor([[1, 0], [0, 0]], dtype=torch.cdouble).unsqueeze(2))
        self.register_buffer("b", torch.tensor([[0, 1], [0, 0]], dtype=torch.cdouble).unsqueeze(2))
        self.register_buffer("c", torch.tensor([[0, 0], [1, 0]], dtype=torch.cdouble).unsqueeze(2))
        self.register_buffer("d", torch.tensor([[0, 0], [0, 1]], dtype=torch.cdouble).unsqueeze(2))

    def matrices(self, thetas: torch.Tensor) -> torch.Tensor:
        if thetas.ndim == 1:
            thetas = thetas.unsqueeze(1)
        assert thetas.size(0) == 3
        phi, theta, omega = thetas[0, :], thetas[1, :], thetas[2, :]
        batch_size = thetas.size(1)

        t_plus = torch.exp(-1j * (phi + omega) / 2)
        t_minus = torch.exp(-1j * (phi - omega) / 2)
        sin_t = torch.sin(theta / 2).unsqueeze(0).unsqueeze(1).repeat((2, 2, 1))
        cos_t = torch.cos(theta / 2).unsqueeze(0).unsqueeze(1).repeat((2, 2, 1))

        a = self.a.repeat(1, 1, batch_size) * cos_t * t_plus
        b = self.b.repeat(1, 1, batch_size) * sin_t * torch.conj(t_minus)
        c = self.c.repeat(1, 1, batch_size) * sin_t * t_minus
        d = self.d.repeat(1, 1, batch_size) * cos_t * torch.conj(t_plus)
        return a - b + c + d

    def apply(self, matrices: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = matrices.size(-1)
        return _apply_batch_gate(state, matrices, self.qubits, self.n_qubits, batch_size)

    def forward(self, state: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: batched state
            thetas: Tensor of size (3, batch_size) which contais the values of `phi`/`theta`/`omega`

        Returns:
            torch.Tensor: the resulting state after applying the gate
        """
        mats = self.matrices(thetas)
        return self.apply(mats, state)

    def extra_repr(self) -> str:
        return f"qubits={self.qubits}, n_qubits={self.n_qubits}"


class ControlledRotationGate(Module):
    n_params = 1

    def __init__(self, gate: str, qubits: ArrayLike, n_qubits: int):
        super().__init__()
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.gate = gate
        self.register_buffer("imat", OPERATIONS_DICT["I"])
        self.register_buffer("paulimat", OPERATIONS_DICT[gate])

    def matrices(self, thetas: torch.Tensor) -> torch.Tensor:
        theta = thetas.squeeze(0) if thetas.ndim == 2 else thetas
        batch_size = len(theta)
        return rot_matrices(theta, self.paulimat, self.imat, batch_size)

    def apply(self, matrices: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = matrices.size(-1)
        controlled_mats = create_controlled_batch_from_operation(matrices, batch_size)
        return _apply_batch_gate(state, controlled_mats, self.qubits, self.n_qubits, batch_size)

    def forward(self, state: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        mats = self.matrices(thetas)
        return self.apply(mats, state)

    def extra_repr(self) -> str:
        return f"qubits={self.qubits}, n_qubits={self.n_qubits}"


class RX(RotationGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__("X", qubits, n_qubits)


class RY(RotationGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__("Y", qubits, n_qubits)


class RZ(RotationGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__("Z", qubits, n_qubits)


class CRX(ControlledRotationGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__("X", qubits, n_qubits)


class CRY(ControlledRotationGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__("Y", qubits, n_qubits)


class CRZ(ControlledRotationGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__("Z", qubits, n_qubits)


class CPHASE(Module):
    n_params = 1

    def __init__(self, qubits: ArrayLike, n_qubits: int):
        super().__init__()
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.register_buffer("imat", torch.eye(4, dtype=torch.cdouble))

    def matrices(self, thetas: torch.Tensor) -> torch.Tensor:
        # NOTE: thetas are assumed to be of shape (1,batch_size) or (batch_size,) because we
        # want to allow e.g. (3,batch_size) in the U gate.
        theta = thetas.squeeze(0) if thetas.ndim == 2 else thetas
        batch_size = len(theta)
        mat = self.imat.repeat((batch_size, 1, 1))
        mat = torch.permute(mat, (1, 2, 0))
        phase_rotation_angles = torch.exp(torch.tensor(1j) * theta).unsqueeze(0).unsqueeze(1)
        mat[3, 3, :] = phase_rotation_angles
        return mat

    def apply(self, matrices: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = matrices.size(-1)
        return _apply_batch_gate(state, matrices, self.qubits, self.n_qubits, batch_size)

    def forward(self, state: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        mats = self.matrices(thetas)
        return self.apply(mats, state)

    def extra_repr(self) -> str:
        return f"qubits={self.qubits}, n_qubits={self.n_qubits}"
