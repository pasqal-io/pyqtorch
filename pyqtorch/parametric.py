from __future__ import annotations

import math

import torch

from pyqtorch.apply import _apply_batch_gate, _vmap_apply_gate
from pyqtorch.matrices import (
    DEFAULT_MATRIX_DTYPE,
    OPERATIONS_DICT,
    _dagger,
    _jacobian,
    _unitary,
    make_controlled,
)
from pyqtorch.primitive import Primitive
from pyqtorch.utils import ApplyFn

APPLY_FN_DICT = {ApplyFn.VMAP: _vmap_apply_gate, ApplyFn.EINSUM: _apply_batch_gate}
DEFAULT_APPLY_FN = ApplyFn.EINSUM


class Parametric(Primitive):
    n_params = 1

    def __init__(
        self,
        gate: str,
        target: int,
        param_name: str = "",
        apply_fn_type: ApplyFn = DEFAULT_APPLY_FN,
    ):
        super().__init__(OPERATIONS_DICT[gate], target)
        self.register_buffer("identity", OPERATIONS_DICT["I"])
        self.apply_fn = APPLY_FN_DICT[apply_fn_type]
        self.param_name = param_name

    def apply_operator(self, operator: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.apply_fn(
            state, operator, self.qubit_support, len(state.size()) - 1, state.size(-1)
        )

    def unitary(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        thetas = values[self.param_name]
        batch_size = len(thetas)
        return _unitary(thetas, self.pauli, self.identity, batch_size)

    def dagger(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return _dagger(self.unitary(values))

    def jacobian(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        thetas = values[self.param_name]
        batch_size = len(thetas)
        return _jacobian(thetas, self.pauli, self.identity, batch_size)

    def apply_jacobian(self, state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.apply_operator(self.jacobian(values), state)


class RX(Parametric):
    def __init__(
        self,
        target: int,
        param_name: str = "",
        apply_fn_type: ApplyFn = DEFAULT_APPLY_FN,
    ):
        super().__init__("X", target, param_name, apply_fn_type)


class RY(Parametric):
    def __init__(
        self,
        target: int,
        param_name: str = "",
        apply_fn_type: ApplyFn = DEFAULT_APPLY_FN,
    ):
        super().__init__("Y", target, param_name, apply_fn_type)


class RZ(Parametric):
    def __init__(self, target: int, param_name: str, apply_fn_type: ApplyFn = DEFAULT_APPLY_FN):
        super().__init__("Z", target, param_name, apply_fn_type)


class PHASE(Parametric):
    def __init__(self, target: int, param_name: str, apply_fn_type: ApplyFn = DEFAULT_APPLY_FN):
        super().__init__("I", target, param_name)
        self.apply_fn = APPLY_FN_DICT[apply_fn_type]
        self.param_name = param_name

    def unitary(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        thetas = values[self.param_name]
        theta = thetas.squeeze(0) if thetas.ndim == 2 else thetas
        batch_mat = self.identity.unsqueeze(2).repeat(1, 1, len(theta))
        batch_mat[1, 1, :] = torch.exp(1.0j * thetas).unsqueeze(0).unsqueeze(1)
        return batch_mat

    def jacobian(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        thetas = values[self.param_name]
        theta = thetas.squeeze(0) if thetas.ndim == 2 else thetas
        batch_mat = (
            torch.zeros((2, 2), dtype=torch.complex128).unsqueeze(2).repeat(1, 1, len(theta))
        )
        batch_mat[1, 1, :] = 1j * torch.exp(1j * thetas).unsqueeze(0).unsqueeze(1)
        return batch_mat


class ControlledRotationGate(Parametric):
    n_params = 1

    def __init__(
        self,
        gate: str,
        control: int | list[int],
        target: int,
        param_name: str = "",
        apply_fn_type: ApplyFn = DEFAULT_APPLY_FN,
    ):
        control = [control] if isinstance(control, int) else control
        self.control = control
        super().__init__(gate, target, param_name, apply_fn_type)
        self.qubit_support = self.control + [self.target]

    def unitary(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        thetas = values[self.param_name]
        batch_size = len(thetas)
        mat = _unitary(thetas, self.pauli, self.identity, batch_size)
        return make_controlled(
            mat, batch_size, len(self.control) - (int)(math.log2(mat.shape[0])) + 1
        )

    def jacobian(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        thetas = values[self.param_name]
        batch_size = len(thetas)
        n_control = len(self.control)
        jU = _jacobian(thetas, self.pauli, self.identity, batch_size)
        n_dim = 2 ** (n_control + 1)
        jC = (
            torch.zeros((n_dim, n_dim), dtype=torch.complex128)
            .unsqueeze(2)
            .repeat(1, 1, batch_size)
        )
        unitary_idx = 2 ** (n_control + 1) - 2
        jC[unitary_idx:, unitary_idx:, :] = jU
        return jC


class CRX(ControlledRotationGate):
    def __init__(
        self,
        control: int | list[int],
        target: int,
        param_name: str,
        apply_fn_type: ApplyFn = DEFAULT_APPLY_FN,
    ):
        super().__init__("X", control, target, param_name, apply_fn_type)


class CRY(ControlledRotationGate):
    def __init__(
        self,
        control: int | list[int],
        target: int,
        param_name: str = "",
        apply_fn_type: ApplyFn = DEFAULT_APPLY_FN,
    ):
        super().__init__("Y", control, target, param_name, apply_fn_type)


class CRZ(ControlledRotationGate):
    def __init__(
        self,
        control: list[int],
        target: int,
        param_name: str = "",
        apply_fn_type: ApplyFn = DEFAULT_APPLY_FN,
    ):
        super().__init__("Z", control, target, param_name, apply_fn_type)


class CPHASE(ControlledRotationGate):
    n_params = 1

    def __init__(
        self,
        control: int | list[int],
        target: int,
        param_name: str,
        apply_fn_type: ApplyFn = DEFAULT_APPLY_FN,
    ):
        super().__init__("S", control, target, param_name, apply_fn_type)
        self.apply_fn = APPLY_FN_DICT[apply_fn_type]
        self.phase = PHASE(target, param_name)

    def unitary(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        thetas = values[self.phase.param_name]
        batch_size = len(thetas)
        return make_controlled(self.phase.unitary(values), batch_size, len(self.control))

    def jacobian(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        thetas = values[self.param_name]
        batch_size = len(thetas)
        n_control = len(self.control)
        jU = self.phase.jacobian(values)
        n_dim = 2 ** (n_control + 1)
        jC = (
            torch.zeros((n_dim, n_dim), dtype=torch.complex128)
            .unsqueeze(2)
            .repeat(1, 1, batch_size)
        )
        unitary_idx = 2 ** (n_control + 1) - 2
        jC[unitary_idx:, unitary_idx:, :] = jU
        return jC


class U(Parametric):
    n_params = 3

    def __init__(
        self, target: int, param_names: list[str], apply_fn_type: ApplyFn = DEFAULT_APPLY_FN
    ):
        self.param_names = param_names
        super().__init__("X", target, param_name="", apply_fn_type=apply_fn_type)

        self.register_buffer(
            "a", torch.tensor([[1, 0], [0, 0]], dtype=DEFAULT_MATRIX_DTYPE).unsqueeze(2)
        )
        self.register_buffer(
            "b", torch.tensor([[0, 1], [0, 0]], dtype=DEFAULT_MATRIX_DTYPE).unsqueeze(2)
        )
        self.register_buffer(
            "c", torch.tensor([[0, 0], [1, 0]], dtype=DEFAULT_MATRIX_DTYPE).unsqueeze(2)
        )
        self.register_buffer(
            "d", torch.tensor([[0, 0], [0, 1]], dtype=DEFAULT_MATRIX_DTYPE).unsqueeze(2)
        )

    def unitary(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        phi = values[self.param_names[0]]
        theta = values[self.param_names[1]]
        omega = values[self.param_names[2]]
        batch_size = 1

        t_plus = torch.exp(-1j * (phi + omega) / 2)
        t_minus = torch.exp(-1j * (phi - omega) / 2)
        sin_t = torch.sin(theta / 2).unsqueeze(0).unsqueeze(1).repeat((2, 2, 1))
        cos_t = torch.cos(theta / 2).unsqueeze(0).unsqueeze(1).repeat((2, 2, 1))

        a = self.a.repeat(1, 1, batch_size) * cos_t * t_plus
        b = self.b.repeat(1, 1, batch_size) * sin_t * torch.conj(t_minus)
        c = self.c.repeat(1, 1, batch_size) * sin_t * t_minus
        d = self.d.repeat(1, 1, batch_size) * cos_t * torch.conj(t_plus)
        return a - b + c + d
