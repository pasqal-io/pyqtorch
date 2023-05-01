from __future__ import annotations

from typing import Any

import torch
from numpy.typing import ArrayLike
from torch.nn import Module

from pyqtorch.core.batched_operation import (
    _apply_batch_gate,
    create_controlled_batch_from_operation,
)
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

    def matrices(self, thetas: dict[str, torch.Tensor]) -> torch.Tensor:
        theta = thetas[self.param_name]
        batch_size = len(theta)
        return rot_matrices(theta, self.paulimat, self.imat, batch_size)

    def apply(self, matrices: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = matrices.size(-1)
        return _apply_batch_gate(state, matrices, self.qubits, self.n_qubits, batch_size)

    def forward(self, thetas: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        mats = self.matrices(thetas)
        return self.apply(mats, state)

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


class MultiParamRotationGate(Module):
    def __init__(self, gate: str, qubits: ArrayLike, n_qubits: int, param_names: list[str]):
        super().__init__()
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.param_names = param_names
        self.gate = gate

    def matrices(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        params_values: list[torch.Tensor] = [params[p_name] for p_name in self.param_names]
        assert all(tensor.size(0) == params_values[0].size(0) for tensor in params_values)
        batch_size = params_values[0].size(0)
        return multiparam_rot_matrices(*tuple(params_values + [batch_size]))

    def apply(self, matrices: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = matrices.size(-1)
        return _apply_batch_gate(state, matrices, self.qubits, self.n_qubits, batch_size)

    def forward(self, params: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        mats = self.matrices(params)
        return self.apply(mats, state)

    def extra_repr(self) -> str:
        return f"'{self.gate}', {self.qubits}, {self.n_qubits}, '{self.param_names}'"


def multiparam_rot_matrices(
    phi: torch.Tensor, theta: torch.Tensor, omega: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """Parametrized arbitrary rotation along the axes of the Bloch sphere

    The angles `phi, theta, omega` in tensor format, applied as:
    U(phi, theta, omega) = RZ(omega)RY(theta)RZ(phi)

    Args:
        phi (torch.Tensor): 1D-tensor holding the values of the `phi` parameter
        theta (torch.Tensor): 1D-tensor holding the values of the `theta` parameter
        omega (torch.Tensor): 1D-tensor holding the values of the `omega` parameter

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """
    t_plus = torch.exp(-1j * (phi + omega) / 2)
    t_minus = torch.exp(-1j * (phi - omega) / 2)
    sin_t = torch.sin(theta / 2).unsqueeze(0).unsqueeze(1)
    sin_t = sin_t.repeat((2, 2, 1))
    cos_t = torch.cos(theta / 2).unsqueeze(0).unsqueeze(1)
    cos_t = cos_t.repeat((2, 2, 1))
    return (
        torch.tensor([[1, 0], [0, 0]], dtype=torch.cdouble).unsqueeze(2).repeat(1, 1, batch_size)
        * cos_t
        * t_plus
        - torch.tensor([[0, 1], [0, 0]], dtype=torch.cdouble).unsqueeze(2).repeat(1, 1, batch_size)
        * sin_t
        * torch.conj(t_minus)
        + torch.tensor([[0, 0], [1, 0]], dtype=torch.cdouble).unsqueeze(2).repeat(1, 1, batch_size)
        * sin_t
        * t_minus
        + torch.tensor([[0, 0], [0, 1]], dtype=torch.cdouble).unsqueeze(2).repeat(1, 1, batch_size)
        * cos_t
        * torch.conj(t_plus)
    )


class ControlledRotationGate(Module):
    def __init__(self, gate: str, qubits: ArrayLike, n_qubits: int, param_name: str):
        super().__init__()
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.param_name = param_name
        self.gate = gate
        self.register_buffer("imat", OPERATIONS_DICT["I"])
        self.register_buffer("paulimat", OPERATIONS_DICT[gate])

    def matrices(self, thetas: dict[str, torch.Tensor]) -> torch.Tensor:
        theta = thetas[self.param_name]
        batch_size = len(theta)
        return rot_matrices(theta, self.paulimat, self.imat, batch_size)

    def apply(self, matrices: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = matrices.size(-1)
        controlled_mats = create_controlled_batch_from_operation(matrices, batch_size)
        return _apply_batch_gate(state, controlled_mats, self.qubits, self.n_qubits, batch_size)

    def forward(self, thetas: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        mats = self.matrices(thetas)
        return self.apply(mats, state)


def RX(*args: Any, **kwargs: Any) -> RotationGate:
    return RotationGate("X", *args, **kwargs)


def RY(*args: Any, **kwargs: Any) -> RotationGate:
    return RotationGate("Y", *args, **kwargs)


def RZ(*args: Any, **kwargs: Any) -> RotationGate:
    return RotationGate("Z", *args, **kwargs)


def U(*args: Any, **kwargs: Any) -> MultiParamRotationGate:
    return MultiParamRotationGate("U", *args, **kwargs)


def CRX(*args: Any, **kwargs: Any) -> RotationGate:
    return ControlledRotationGate("X", *args, **kwargs)


def CRY(*args: Any, **kwargs: Any) -> RotationGate:
    return ControlledRotationGate("Y", *args, **kwargs)


def CRZ(*args: Any, **kwargs: Any) -> RotationGate:
    return ControlledRotationGate("Z", *args, **kwargs)


class CPHASE(Module):
    def __init__(self, qubits: ArrayLike, n_qubits: int, param_name: str):
        super().__init__()
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.param_name = param_name
        self.register_buffer("imat", torch.eye(4, dtype=torch.cdouble))

    def forward(self, thetas: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        theta = thetas[self.param_name]
        batch_size = len(theta)
        mat = self.imat.repeat((batch_size, 1, 1))
        mat = torch.permute(mat, (1, 2, 0))
        phase_rotation_angles = torch.exp(torch.tensor(1j) * theta).unsqueeze(0).unsqueeze(1)
        mat[3, 3, :] = phase_rotation_angles
        return _apply_batch_gate(state, mat, self.qubits, self.n_qubits, batch_size)
