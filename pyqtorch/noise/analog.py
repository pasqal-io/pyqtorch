from __future__ import annotations

from math import log, sqrt
from typing import Any, Callable

import torch
from torch import Tensor

from pyqtorch.matrices import XMAT, YMAT, ZMAT
from pyqtorch.qubit_support import Support
from pyqtorch.time_dependent.mesolve import mesolve
from pyqtorch.utils import (
    DensityMatrix,
    SolverType,
    density_mat,
    expand_operator,
    permute_basis,
)


class AnalogNoise(torch.nn.Module):
    """AnalogNoise is used within `HamiltonianEvolution`
        when using a Shrodinger equation solver.

    Attributes:
        noise_operators (list[Tensor]): The list of jump operators
            to use with mesolve. Note you can only define them as
            2D tensors, so no batchsize.
        qubit_support (int | tuple[int, ...] | Support): The qubits
            where the noise_operators are defined over.

    """

    def __init__(
        self,
        noise_operators: list[Tensor],
        qubit_support: int | tuple[int, ...] | Support,
    ) -> None:
        super().__init__()
        self._qubit_support = (
            qubit_support
            if isinstance(qubit_support, Support)
            else Support(target=qubit_support)
        )

        for index, tensor in enumerate(noise_operators):
            if len(self._qubit_support) < int(log(tensor.shape[1], 2)):
                raise ValueError(
                    "Tensor dimensions should match the length of the qubit support."
                )
            self.register_buffer(f"noise_operators_{index}", tensor)

        self._device: torch.device = noise_operators[0].device
        self._dtype: torch.dtype = noise_operators[0].dtype

    @property
    def noise_operators(self) -> list[Tensor]:
        return [
            getattr(self, f"noise_operators_{i}") for i in range(len(self._buffers))
        ]

    @property
    def qubit_support(self) -> tuple[int, ...]:
        """Getter qubit_support.

        Returns:
            Support: Tuple of sorted qubits.
        """
        return self._qubit_support.sorted_qubits

    def extra_repr(self) -> str:
        return f"noise_operators: {self.noise_operators}"

    def _noise_operators(
        self,
        full_support: tuple[int, ...] | None = None,
    ) -> list[Tensor]:
        """Obtain noise operators expended with full support.

        Args:
            full_support (tuple[int, ...] | None, optional): _description_. Defaults to None.

        Returns:
            list[Tensor]: _description_
        """
        list_ops = self.noise_operators
        if self._qubit_support.qubits != self.qubit_support:
            list_ops = [
                permute_basis(
                    blockmat.unsqueeze(-1), self._qubit_support.qubits, inv=True
                ).squeeze(-1)
                for blockmat in list_ops
            ]
        if full_support is None:
            return list_ops
        else:
            return [
                expand_operator(
                    blockmat.unsqueeze(-1), self.qubit_support, full_support
                ).squeeze(-1)
                for blockmat in list_ops
            ]

    def to(self, *args: Any, **kwargs: Any) -> AnalogNoise:
        super().to(*args, **kwargs)
        self._device = self.noise_operators_0.device
        self._dtype = self.noise_operators_0.dtype
        return self

    def forward(
        self,
        state: Tensor,
        H: Callable[..., Any],
        tsave: list | Tensor,
        solver: SolverType,
        options: dict[str, Any] | None = None,
        full_support: tuple[int, ...] | None = None,
    ) -> Tensor:
        if not isinstance(state, DensityMatrix):
            state = density_mat(state)
        sol = mesolve(
            H,
            state,
            self._noise_operators(full_support),
            tsave,
            solver,
            options=options,
        )
        return DensityMatrix(sol.states[-1])


class Depolarizing(AnalogNoise):
    def __init__(
        self,
        error_param: float,
        qubit_support: int | tuple[int, ...] | Support,
    ) -> None:
        coeff = sqrt(error_param / 4)
        L0 = coeff * XMAT.squeeze()
        L1 = coeff * YMAT.squeeze()
        L2 = coeff * ZMAT.squeeze()
        noise_operators: list[Tensor] = [L0, L1, L2]
        super().__init__(noise_operators, qubit_support)
