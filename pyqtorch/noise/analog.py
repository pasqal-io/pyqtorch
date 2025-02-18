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
        when solving the Schrödinger or a master (Lindblad) equation.

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
                    f"Tensor {tensor} has incompatible dimensions {int(log(tensor.shape[1], 2))}."
                )
            self.register_buffer(f"noise_operators_{index}", tensor)

        self._device: torch.device = noise_operators[0].device
        self._dtype: torch.dtype = noise_operators[0].dtype

    @property
    def noise_operators(self) -> list[Tensor]:
        """Get noise_operators.

        Returns:
            list[Tensor]: Noise operators.
        """
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
            list[Tensor]: Noise operators defined over `full_support`.
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
        """Do device or dtype conversions.

        Returns:
            AnalogNoise: Converted instance.
        """
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
        """Obtain the output density matrix by solving a Shrodinger equation.

        This uses the `mesolve` function.

        Args:
            state (Tensor): Input state or density matrix.
            H (Callable[..., Any]): Time-dependent hamiltonian fonction.
            tsave (list | Tensor): Tensor containing simulation time instants.
            solver (SolverType): Name of the solver to use.
            options (dict[str, Any] | None, optional): Additional options passed to the solver.
                Defaults to None.
            full_support (tuple[int, ...] | None, optional): The qubits the returned tensor
                will be defined over. Defaults to None for only using the qubit_support.

        Returns:
            Tensor: Output density matrix from solver.
        """
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

    def __add__(self, other: AnalogNoise) -> AnalogNoise:
        return AnalogNoise(
            self.noise_operators + other.noise_operators,
            self._qubit_support + other._qubit_support,
        )


class AnalogDepolarizing(AnalogNoise):
    """
    Defines jump operators for an Analog Depolarizing channel.

    Under the depolarizing noise, a system in any state evolves
      to the maximally mixed state at a rate :math:`p`.

    The corresponding jump operators are :
    .. math::
        `L_{0,1,2} = \\sqrt{\\frac{p}{4}} \\sigma_{x,y,z}`

    where :math:`\\sigma_{x,y,z}` correspond to the unitaries of the X,Y,Z gates.

        Args:
            error_param (float): Rate of depolarizing.
            qubit_support (int | tuple[int, ...] | Support): Qubits defining the operation.
    """

    def __init__(
        self,
        error_param: float,
        qubit_support: int | tuple[int, ...] | Support,
    ) -> None:
        """Initializes AnalogDepolarizing.

        Args:
            error_param (float): Rate of depolarizing.
            qubit_support (int | tuple[int, ...] | Support): Qubits defining the operation.
        """
        coeff = sqrt(error_param / 4)
        L0 = coeff * XMAT.squeeze()
        L1 = coeff * YMAT.squeeze()
        L2 = coeff * ZMAT.squeeze()
        noise_operators: list[Tensor] = [L0, L1, L2]
        super().__init__(noise_operators, qubit_support)
