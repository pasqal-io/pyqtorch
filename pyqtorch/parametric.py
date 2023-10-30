from __future__ import annotations

from math import log2
from typing import Tuple

import torch

from pyqtorch.matrices import (
    DEFAULT_MATRIX_DTYPE,
    OPERATIONS_DICT,
    _controlled,
    _jacobian,
    _unitary,
)
from pyqtorch.primitive import Primitive
from pyqtorch.utils import Operator


class Parametric(Primitive):
    n_params = 1

    def __init__(
        self,
        generator_name: str,
        target: int,
        param_name: str = "",
    ):
        super().__init__(OPERATIONS_DICT[generator_name], target)
        self.register_buffer("identity", OPERATIONS_DICT["I"])
        self.param_name = param_name

        def parse_values(values: dict[str, torch.Tensor] | torch.Tensor = {}) -> torch.Tensor:
            return Parametric._expand_values(values[self.param_name])

        def parse_tensor(values: dict[str, torch.Tensor] | torch.Tensor = {}) -> torch.Tensor:
            return Parametric._expand_values(values)

        self.parse_values = parse_tensor if param_name == "" else parse_values

    @staticmethod
    def _expand_values(values: torch.Tensor) -> torch.Tensor:
        return values.unsqueeze(0) if len(values.size()) == 0 else values

    def unitary(self, values: dict[str, torch.Tensor] | torch.Tensor = {}) -> Operator:
        thetas = self.parse_values(values)
        batch_size = len(thetas)
        return _unitary(thetas, self.pauli, self.identity, batch_size)

    def jacobian(self, values: dict[str, torch.Tensor] | torch.Tensor = {}) -> Operator:
        thetas = self.parse_values(values)
        batch_size = len(thetas)
        return _jacobian(thetas, self.pauli, self.identity, batch_size)


class RX(Parametric):
    def __init__(
        self,
        target: int,
        param_name: str = "",
    ):
        super().__init__("X", target, param_name)


class RY(Parametric):
    def __init__(
        self,
        target: int,
        param_name: str = "",
    ):
        super().__init__("Y", target, param_name)


class RZ(Parametric):
    def __init__(self, target: int, param_name: str = ""):
        super().__init__("Z", target, param_name)


class PHASE(Parametric):
    def __init__(self, target: int, param_name: str = ""):
        super().__init__("I", target, param_name)

    def unitary(self, values: dict[str, torch.Tensor] = {}) -> Operator:
        thetas = self.parse_values(values)
        batch_size = len(thetas)
        batch_mat = self.identity.unsqueeze(2).repeat(1, 1, batch_size)
        batch_mat[1, 1, :] = torch.exp(1.0j * thetas).unsqueeze(0).unsqueeze(1)
        return batch_mat

    def jacobian(self, values: dict[str, torch.Tensor] = {}) -> Operator:
        thetas = self.parse_values(values)
        batch_mat = (
            torch.zeros((2, 2), dtype=torch.complex128).unsqueeze(2).repeat(1, 1, len(thetas))
        )
        batch_mat[1, 1, :] = 1j * torch.exp(1j * thetas).unsqueeze(0).unsqueeze(1)
        return batch_mat


class ControlledRotationGate(Parametric):
    n_params = 1

    def __init__(
        self,
        gate: str,
        control: int | Tuple[int, ...],
        target: int,
        param_name: str = "",
    ):
        self.control = control if isinstance(control, tuple) else (control,)
        super().__init__(gate, target, param_name)
        self.qubit_support = self.control + (self.target,)
        self.n_qubits = max(list(self.qubit_support))

    def unitary(self, values: dict[str, torch.Tensor] = {}) -> Operator:
        thetas = self.parse_values(values)
        batch_size = len(thetas)
        mat = _unitary(thetas, self.pauli, self.identity, batch_size)
        return _controlled(mat, batch_size, len(self.control) - (int)(log2(mat.shape[0])) + 1)

    def jacobian(self, values: dict[str, torch.Tensor] = {}) -> Operator:
        thetas = self.parse_values(values)
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
        control: int | Tuple[int],
        target: int,
        param_name: str = "",
    ):
        super().__init__("X", control, target, param_name)


class CRY(ControlledRotationGate):
    def __init__(
        self,
        control: int | Tuple[int],
        target: int,
        param_name: str = "",
    ):
        super().__init__("Y", control, target, param_name)


class CRZ(ControlledRotationGate):
    def __init__(
        self,
        control: Tuple[int],
        target: int,
        param_name: str = "",
    ):
        super().__init__("Z", control, target, param_name)


class CPHASE(ControlledRotationGate):
    n_params = 1

    def __init__(
        self,
        control: int | Tuple[int],
        target: int,
        param_name: str = "",
    ):
        super().__init__("S", control, target, param_name)
        self.phase = PHASE(target, param_name)

    def unitary(self, values: dict[str, torch.Tensor] = {}) -> Operator:
        thetas = self.parse_values(values)
        batch_size = len(thetas)
        return _controlled(self.phase.unitary(values), batch_size, len(self.control))

    def jacobian(self, values: dict[str, torch.Tensor] = {}) -> Operator:
        thetas = self.parse_values(values)
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

    def __init__(self, target: int, phi: str, theta: str, omega: str):
        self.phi = phi
        self.theta = theta
        self.omega = omega
        super().__init__("X", target, param_name="")

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

    def unitary(self, values: dict[str, torch.Tensor] = {}) -> Operator:
        phi, theta, omega = list(
            map(
                lambda t: t.unsqueeze(0) if len(t.size()) == 0 else t,
                [values[self.phi], values[self.theta], values[self.omega]],
            )
        )
        batch_size = len(theta)

        t_plus = torch.exp(-1j * (phi + omega) / 2)
        t_minus = torch.exp(-1j * (phi - omega) / 2)
        sin_t = torch.sin(theta / 2).unsqueeze(0).unsqueeze(1).repeat((2, 2, 1))
        cos_t = torch.cos(theta / 2).unsqueeze(0).unsqueeze(1).repeat((2, 2, 1))

        a = self.a.repeat(1, 1, batch_size) * cos_t * t_plus
        b = self.b.repeat(1, 1, batch_size) * sin_t * torch.conj(t_minus)
        c = self.c.repeat(1, 1, batch_size) * sin_t * t_minus
        d = self.d.repeat(1, 1, batch_size) * cos_t * torch.conj(t_plus)
        return a - b + c + d

    def jacobian(self, values: dict[str, torch.Tensor] = {}) -> Operator:
        raise NotImplementedError

    def digital_decomposition(self) -> list[Parametric]:
        return [
            RZ(self.qubit_support[0], self.phi),
            RY(self.qubit_support[0], self.theta),
            RZ(self.qubit_support[0], self.omega),
        ]

    def jacobian_decomposed(self, values: dict[str, torch.Tensor] = {}) -> list[Operator]:
        return [op.jacobian(values) for op in self.digital_decomposition()]
