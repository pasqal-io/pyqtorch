from __future__ import annotations

from typing import Tuple

import torch

from pyqtorch.matrices import DEFAULT_MATRIX_DTYPE, IMAT, XMAT, _dagger
from pyqtorch.utils import Operator, State


class Noise(torch.nn.Module):
    # * PROBA in tuple
    def __init__(
        self, kraus: Tuple[Operator, ...], target: int, probability: Tuple[float, ...]
    ) -> None:
        super().__init__()
        self.target: int = target
        self.qubit_support: Tuple[int, ...] = (target,)
        self.n_qubits: int = max(self.qubit_support)

        self.probabilty = probability
        # * Put conditions on proba

        self.register_buffer("kraus", kraus)
        self._param_type = None
        self._device = self.kraus.device

    @property
    def probabilty(self) -> float:
        return self.probabilty

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.name == other.name and self.probabilty == other.probabilty
        else:
            return False

    def extra_repr(self) -> str:
        return f"{self.name}('qubit_support'={self.qubit_support}, 'probability'={self.probabilty})"

    @property
    def kraus(self) -> Tuple[Operator, ...]:
        return self.kraus

    @property
    def param_type(self) -> None:
        return self._param_type

    @classmethod
    def unitary(cls, kraus_op: Tuple[torch.tensor], batch_size) -> torch.tensor:
        return kraus_op.unsqueeze(2).repeat(1, 1, batch_size)

    def dagger(self, values: dict[str, torch.Tensor] | torch.Tensor = {}) -> Operator:
        return _dagger(self.unitary(values))

    def forward(self, state: State, kraus) -> State:
        return 1
        # * return A@rho@A_dag

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device) -> Noise:
        super().to(device)
        self._device = device
        return self


class Bitflip(Noise):
    def __init__(
        self,
        target: int,
        probability: float,
    ):
        K0 = torch.sqrt(torch.tensor(1 - self.probability[0], dtype=DEFAULT_MATRIX_DTYPE)) * IMAT
        K1 = torch.sqrt(torch.tensor(self.probability[0], dtype=DEFAULT_MATRIX_DTYPE)) * XMAT
        Kraus_Bitflip = [K0, K1]
        super().__init__(Kraus_Bitflip, target, probability)
