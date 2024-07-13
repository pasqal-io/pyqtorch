from __future__ import annotations

import logging
from functools import cached_property
from logging import getLogger
from typing import Any

import numpy as np
import torch
from numpy import log2
from torch import Tensor

from pyqtorch.apply import apply_density_mat, apply_operator, operator_product
from pyqtorch.matrices import (
    IMAT,
    OPERATIONS_DICT,
    _controlled,
    _dagger,
)
from pyqtorch.noise import Noisy_protocols
from pyqtorch.utils import DensityMatrix, density_mat, product_state

logger = getLogger(__name__)


def forward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    torch.cuda.nvtx.range_pop()


def pre_forward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    torch.cuda.nvtx.range_push("Primitive.forward")


def backward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    torch.cuda.nvtx.range_pop()


def pre_backward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    torch.cuda.nvtx.range_push("Primitive.backward")


class Primitive(torch.nn.Module):
    def __init__(
        self,
        pauli: Tensor,
        target: int | tuple[int, ...],
        noise: Noisy_protocols | dict[str, Noisy_protocols] | None = None,
    ) -> None:
        super().__init__()
        self.target: int | tuple[int, ...] = target
        self.noise: Noisy_protocols | dict[str, Noisy_protocols] | None = noise
        self.qubit_support: tuple[int, ...] = (
            (target,) if isinstance(target, int) else target
        )
        if isinstance(target, np.integer):
            self.qubit_support = (target.item(),)
        self.register_buffer("pauli", pauli)
        self._device = self.pauli.device
        self._dtype = self.pauli.dtype

        if logger.isEnabledFor(logging.DEBUG):
            # When Debugging let's add logging and NVTX markers
            # WARNING: incurs performance penalty
            self.register_forward_hook(forward_hook, always_call=True)
            self.register_full_backward_hook(backward_hook)
            self.register_forward_pre_hook(pre_forward_hook)
            self.register_full_backward_pre_hook(pre_backward_hook)

    def __hash__(self) -> int:
        return hash(self.qubit_support)

    def extra_repr(self) -> str:
        if self.noise:
            noise_info = ""
            if isinstance(self.noise, Noisy_protocols):
                noise_info = str(self.noise)
            elif isinstance(self.noise, dict):
                noise_info = ", ".join(
                    str(noise_instance) for noise_instance in self.noise.values()
                )
            return f"target: {self.qubit_support}, Noise: {noise_info}"
        return f"target: {self.qubit_support}"

    def unitary(self, values: dict[str, Tensor] | Tensor = dict()) -> Tensor:
        return self.pauli.unsqueeze(2) if len(self.pauli.shape) == 2 else self.pauli

    def forward(
        self, state: Tensor, values: dict[str, Tensor] | Tensor = dict()
    ) -> Tensor:
        if self.noise:
            if not isinstance(state, DensityMatrix):
                state = density_mat(state)
            n_qubits = int(log2(state.size(1)))
            state = apply_density_mat(self.tensor(values, n_qubits), state)
            if isinstance(self.noise, dict):
                for noise_instance in self.noise.values():
                    protocol = noise_instance.protocol_to_gate()
                    noise_gate = protocol(
                        target=(
                            noise_instance.target
                            if noise_instance.target is not None
                            else self.target
                        ),
                        error_probability=noise_instance.error_probability,
                    )
                    state = noise_gate(state, values)
                return state
            else:
                protocol = self.noise.protocol_to_gate()
                noise_gate = protocol(
                    target=(
                        self.noise.target
                        if self.noise.target is not None
                        else self.target
                    ),
                    error_probability=self.noise.error_probability,
                )
                return noise_gate(state, values)
        else:
            if isinstance(state, DensityMatrix):
                n_qubits = int(log2(state.size(1)))
                return apply_density_mat(self.tensor(values, n_qubits), state)
            else:
                return apply_operator(
                    state,
                    self.unitary(values),
                    self.qubit_support,
                    len(state.size()) - 1,
                )

    # ? Do we need to keep this method now ?
    def dagger(self, values: dict[str, Tensor] | Tensor = dict()) -> Tensor:
        return _dagger(self.unitary(values))

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def to(self, *args: Any, **kwargs: Any) -> Primitive:
        super().to(*args, **kwargs)
        self._device = self.pauli.device
        self._dtype = self.pauli.dtype
        return self

    @cached_property
    def eigenvals_generator(self) -> Tensor:
        return torch.linalg.eigvalsh(self.pauli).reshape(-1, 1)

    @cached_property
    def spectral_gap(self) -> Tensor:
        spectrum = self.eigenvals_generator
        spectral_gap = torch.unique(torch.abs(torch.tril(spectrum - spectrum.T)))
        return spectral_gap[spectral_gap.nonzero()]

    def tensor(
        self, values: dict[str, Tensor] = {}, n_qubits: int = 1, diagonal: bool = False
    ) -> Tensor:
        if diagonal:
            raise NotImplementedError
        blockmat = self.unitary(values)
        if n_qubits == 1:
            return blockmat
        full_sup = tuple(i for i in range(n_qubits))
        support = tuple(sorted(self.qubit_support))
        mat = (
            IMAT.clone().to(self.device).unsqueeze(2)
            if support[0] != full_sup[0]
            else blockmat
        )
        for i in full_sup[1:]:
            if i == support[0]:
                other = blockmat
                mat = torch.kron(mat.contiguous(), other.contiguous())
            elif i not in support:
                other = IMAT.clone().to(self.device).unsqueeze(2)
                mat = torch.kron(mat.contiguous(), other.contiguous())
        return mat


class X(Primitive):
    def __init__(
        self,
        target: int,
        noise: Noisy_protocols | dict[str, Noisy_protocols] | None = None,
    ):
        super().__init__(OPERATIONS_DICT["X"], target, noise)


class Y(Primitive):
    def __init__(
        self,
        target: int,
        noise: Noisy_protocols | dict[str, Noisy_protocols] | None = None,
    ):
        super().__init__(OPERATIONS_DICT["Y"], target, noise)


class Z(Primitive):
    def __init__(
        self,
        target: int,
        noise: Noisy_protocols | dict[str, Noisy_protocols] | None = None,
    ):
        super().__init__(OPERATIONS_DICT["Z"], target, noise)


class I(Primitive):  # noqa: E742
    def __init__(
        self,
        target: int,
        noise: Noisy_protocols | dict[str, Noisy_protocols] | None = None,
    ):
        super().__init__(OPERATIONS_DICT["I"], target, noise)

    # FIXME: apply_operator is not compatible with the identity tensor
    def forward(self, state: Tensor, values: dict[str, Tensor] = dict()) -> Tensor:
        if self.noise:
            if not isinstance(state, DensityMatrix):
                state = density_mat(state)
            n_qubits = int(log2(state.size(1)))
            state = apply_density_mat(self.tensor(values, n_qubits), state)
            if isinstance(self.noise, dict):
                for noise_instance in self.noise.values():
                    protocol = noise_instance.protocol_to_gate()
                    noise_gate = protocol(
                        target=(
                            noise_instance.target
                            if noise_instance.target is not None
                            else self.target
                        ),
                        error_probability=noise_instance.error_probability,
                    )
                    state = noise_gate(state, values)
                return state
            else:
                protocol = self.noise.protocol_to_gate()
                noise_gate = protocol(
                    target=(
                        self.noise.target
                        if self.noise.target is not None
                        else self.target
                    ),
                    error_probability=self.noise.error_probability,
                )
                return noise_gate(state, values)
        return state


class H(Primitive):
    def __init__(
        self,
        target: int,
        noise: Noisy_protocols | dict[str, Noisy_protocols] | None = None,
    ):
        super().__init__(OPERATIONS_DICT["H"], target, noise)


class T(Primitive):
    def __init__(
        self,
        target: int,
        noise: Noisy_protocols | dict[str, Noisy_protocols] | None = None,
    ):
        super().__init__(OPERATIONS_DICT["T"], target, noise)


class S(Primitive):
    def __init__(
        self,
        target: int,
        noise: Noisy_protocols | dict[str, Noisy_protocols] | None = None,
    ):
        super().__init__(OPERATIONS_DICT["S"], target, noise)


class SDagger(Primitive):
    def __init__(
        self,
        target: int,
        noise: Noisy_protocols | dict[str, Noisy_protocols] | None = None,
    ):
        super().__init__(OPERATIONS_DICT["SDAGGER"], target, noise)


class Projector(Primitive):
    def __init__(
        self,
        qubit_support: int | tuple[int, ...],
        ket: str,
        bra: str,
        noise: Noisy_protocols | dict[str, Noisy_protocols] | None = None,
    ):
        support = (qubit_support,) if isinstance(qubit_support, int) else qubit_support
        if len(ket) != len(bra):
            raise ValueError("Input ket and bra bitstrings must be of same length.")
        ket_state = product_state(ket).flatten()
        bra_state = product_state(bra).flatten()
        super().__init__(
            OPERATIONS_DICT["PROJ"](ket_state, bra_state), support[-1], noise
        )
        # Override the attribute in AbstractOperator.
        self.qubit_support = support


class N(Primitive):
    def __init__(
        self,
        target: int,
        noise: Noisy_protocols | dict[str, Noisy_protocols] | None = None,
    ):
        super().__init__(OPERATIONS_DICT["N"], target, noise)


class SWAP(Primitive):
    def __init__(self, control: int, target: int):
        # TODO: Change the control param name
        super().__init__(OPERATIONS_DICT["SWAP"], target)
        self.control = control
        self.qubit_support = (self.control,) + (target,)

    def tensor(
        self, values: dict[str, Tensor] = {}, n_qubits: int = 1, diagonal: bool = False
    ) -> Tensor:
        from pyqtorch.circuit import Sequence

        if diagonal:
            raise NotImplementedError
        if n_qubits < max(self.qubit_support) + 1:
            n_qubits = max(self.qubit_support) + 1
        seq = Sequence(
            [
                CNOT(control=self.control, target=self.target),  # type: ignore[arg-type]
                CNOT(control=self.target, target=self.control),  # type: ignore[arg-type]
                CNOT(control=self.control, target=self.target),  # type: ignore[arg-type]
            ]
        )
        return seq.tensor(values, n_qubits)


class CSWAP(Primitive):
    def __init__(self, control: int | tuple[int, ...], target: tuple[int, ...]):
        if not isinstance(target, tuple) or len(target) != 2:
            raise ValueError("Target qubits must be a tuple with two qubits")
        super().__init__(OPERATIONS_DICT["CSWAP"], target)
        self.control = (control,) if isinstance(control, int) else control
        self.target = target
        self.qubit_support = self.control + self.target

    def extra_repr(self) -> str:
        return f"control:{self.control}, target:{self.target}"

    def tensor(
        self, values: dict[str, Tensor] = {}, n_qubits: int = 1, diagonal: bool = False
    ) -> Tensor:
        from pyqtorch.circuit import Sequence

        if diagonal:
            raise NotImplementedError
        if n_qubits < max(self.qubit_support) + 1:
            n_qubits = max(self.qubit_support) + 1
        seq = Sequence(
            [
                Toffoli(
                    control=(self.control[0], self.target[1]), target=self.target[0]  # type: ignore[index]
                ),
                Toffoli(
                    control=(self.control[0], self.target[0]), target=self.target[1]  # type: ignore[index]
                ),
                Toffoli(
                    control=(self.control[0], self.target[1]), target=self.target[0]  # type: ignore[index]
                ),
            ]  #! Can't take more that 1 control in this logic
        )
        return seq.tensor(values, n_qubits)


class ControlledOperationGate(Primitive):
    def __init__(
        self,
        gate: str,
        control: int | tuple[int, ...],
        target: int,
        noise: Noisy_protocols | dict[str, Noisy_protocols] | None = None,
    ):
        self.control = (control,) if isinstance(control, int) else control
        mat = OPERATIONS_DICT[gate]
        mat = _controlled(
            unitary=mat.unsqueeze(2),
            batch_size=1,
            n_control_qubits=len(self.control),
        ).squeeze(2)
        super().__init__(mat, target, noise)
        self.gate = globals()[gate]
        self.qubit_support = self.control + (self.target,)  # type: ignore[operator]
        self.noise = noise

    def extra_repr(self) -> str:
        if self.noise:
            noise_info = ""
            if isinstance(self.noise, Noisy_protocols):
                noise_info = str(self.noise)
            elif isinstance(self.noise, dict):
                noise_info = ", ".join(
                    str(noise_instance) for noise_instance in self.noise.values()
                )
            return (
                f"control:{self.control}, target:{(self.target,)}, Noise: {noise_info}"
            )
        return f"control:{self.control}, target:{(self.target,)}"

    def tensor(
        self, values: dict[str, Tensor] = {}, n_qubits: int = 1, diagonal: bool = False
    ) -> Tensor:
        from pyqtorch.circuit import Sequence

        if diagonal:
            raise NotImplementedError
        if n_qubits < max(self.qubit_support) + 1:
            n_qubits = max(self.qubit_support) + 1
        proj1 = Sequence(
            [Projector(qubit_support=qubit, ket="1", bra="1") for qubit in self.control]
        )
        c_mat = (
            I(target=self.control[0]).tensor(values, n_qubits)
            - proj1.tensor(values, n_qubits)
            + operator_product(
                proj1.tensor(values, n_qubits),
                self.gate(self.target).tensor(values, n_qubits),
            )
        )
        return c_mat


class CNOT(ControlledOperationGate):
    def __init__(
        self,
        control: int | tuple[int, ...],
        target: int,
        noise: Noisy_protocols | dict[str, Noisy_protocols] | None = None,
    ):
        super().__init__("X", control, target, noise)


CX = CNOT


class CY(ControlledOperationGate):
    def __init__(
        self,
        control: int | tuple[int, ...],
        target: int,
        noise: Noisy_protocols | dict[str, Noisy_protocols] | None = None,
    ):
        super().__init__("Y", control, target, noise)


class CZ(ControlledOperationGate):
    def __init__(
        self,
        control: int | tuple[int, ...],
        target: int,
        noise: Noisy_protocols | dict[str, Noisy_protocols] | None = None,
    ):
        super().__init__("Z", control, target, noise)


class Toffoli(ControlledOperationGate):
    def __init__(
        self,
        control: tuple[int, ...],
        target: int,
        noise: Noisy_protocols | dict[str, Noisy_protocols] | None = None,
    ):
        super().__init__("X", control, target, noise)
