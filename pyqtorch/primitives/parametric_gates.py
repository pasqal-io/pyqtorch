from __future__ import annotations

from functools import cached_property
from typing import Any

import torch
from torch import Tensor

from pyqtorch.embed import Embedding
from pyqtorch.matrices import (
    DEFAULT_MATRIX_DTYPE,
    controlled,
)
from pyqtorch.noise import DigitalNoiseProtocol

from .parametric import ControlledRotationGate, Parametric

pauli_singleq_eigenvalues = torch.tensor([[-1.0], [1.0]], dtype=torch.cdouble)


class RX(Parametric):
    """
    Primitive for the RX gate.

    The corresponding unitary is obtained by :math:`exp(-i X, t)`
    where :math:`t` is an input parameter
    and :math:`X` is the matrix representation of the X gate.

    Attributes:
        param_name: Name of parameters.
        parse_values: Method defining how to handle the values dictionary input.
    """

    def __init__(
        self,
        target: int,
        param_name: str | int | float | torch.Tensor = "",
        noise: DigitalNoiseProtocol | None = None,
    ):
        """Initializes RX.

        Arguments:
            target: Target qubit.
            param_name: Name of parameters.
            noise: Optional noise protocols to apply.
        """
        super().__init__("X", target, param_name, noise=noise)

    @cached_property
    def eigenvals_generator(self) -> Tensor:
        """Get eigenvalues of the underlying generator.

        Arguments:
            values: Parameter values.

        Returns:
            Eigenvalues of the generator operator.
        """
        return pauli_singleq_eigenvalues.to(device=self.device, dtype=self.dtype)


class RY(Parametric):
    """
    Primitive for the RY gate.

    The corresponding unitary is obtained by :math:`exp(-i Y, t)`
    where :math:`t` is an input parameter
    and :math:`Y` is the matrix representation of the Y gate.

    Attributes:
        param_name: Name of parameters.
        parse_values: Method defining how to handle the values dictionary input.
    """

    def __init__(
        self,
        target: int,
        param_name: str | int | float | torch.Tensor = "",
        noise: DigitalNoiseProtocol | None = None,
    ):
        """Initializes RY.

        Arguments:
            target: Target qubit.
            param_name: Name of parameters.
            noise: Optional noise protocols to apply.
        """
        super().__init__("Y", target, param_name, noise=noise)

    @cached_property
    def eigenvals_generator(self) -> Tensor:
        """Get eigenvalues of the underlying generator.

        Arguments:
            values: Parameter values.

        Returns:
            Eigenvalues of the generator operator.
        """
        return pauli_singleq_eigenvalues.to(device=self.device, dtype=self.dtype)


class RZ(Parametric):
    """
    Primitive for the RZ gate.

    The corresponding unitary is obtained by :math:`exp(-i Z, t)`
    where :math:`t` is an input parameter
    and :math:`Z` is the matrix representation of the Z gate.

    Attributes:
        param_name: Name of parameters.
        parse_values: Method defining how to handle the values dictionary input.
    """

    def __init__(
        self,
        target: int,
        param_name: str | int | float | torch.Tensor = "",
        noise: DigitalNoiseProtocol | None = None,
    ):
        """Initializes RZ.

        Arguments:
            target: Target qubit.
            param_name: Name of parameters.
            noise: Optional noise protocols to apply.
        """
        super().__init__("Z", target, param_name, noise=noise)

    @cached_property
    def eigenvals_generator(self) -> Tensor:
        """Get eigenvalues of the underlying generator.

        Arguments:
            values: Parameter values.

        Returns:
            Eigenvalues of the generator operator.
        """
        return pauli_singleq_eigenvalues.to(device=self.device, dtype=self.dtype)


class PHASE(Parametric):
    """
    Primitive for the PHASE gate.

    The corresponding unitary is obtained by :math:`exp(-i I, t)`
    where :math:`t` is an input parameter
    and :math:`I` is the identity.

    Attributes:
        param_name: Name of parameters.
        parse_values: Method defining how to handle the values dictionary input.
    """

    def __init__(
        self,
        target: int,
        param_name: str | int | float | torch.Tensor = "",
        noise: DigitalNoiseProtocol | None = None,
    ):
        """Initializes PHASE.

        Arguments:
            target: Target qubit.
            param_name: Name of parameters.
            noise: Optional noise protocols to apply.
        """
        super().__init__("I", target, param_name, noise=noise)

    @cached_property
    def eigenvals_generator(self) -> Tensor:
        """Get eigenvalues of the underlying generator.

        Arguments:
            values: Parameter values.

        Returns:
            Eigenvalues of the generator operator.
        """
        return torch.tensor([[0.0], [2.0]], dtype=self.dtype, device=self.device)

    def _construct_parametric_base_op(
        self,
        values: dict[str, Tensor] | None = None,
        embedding: Embedding | None = None,
    ) -> Tensor:
        """
        Get the corresponding unitary.

        Arguments:
            values: A dict containing a Parameter name and value.
            embedding: An optional embedding for parameters.

        Returns:
            The unitary representation.
        """
        values = values or dict()
        thetas = self.parse_values(values, embedding)
        batch_size = len(thetas)
        batch_mat = self.identity.unsqueeze(2).repeat(1, 1, batch_size)
        batch_mat[1, 1, :] = torch.exp(1.0j * thetas).unsqueeze(0).unsqueeze(1)
        return batch_mat

    def jacobian(
        self,
        values: dict[str, Tensor] | None = None,
        embedding: Embedding | None = None,
    ) -> Tensor:
        """
        Get the corresponding unitary of the jacobian.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation of the jacobian.
        """
        values = values or dict()
        thetas = self.parse_values(values, embedding)
        batch_mat = (
            torch.zeros((2, 2), dtype=self.identity.dtype)
            .unsqueeze(2)
            .repeat(1, 1, len(thetas))
        )
        batch_mat[1, 1, :] = 1j * torch.exp(1j * thetas).unsqueeze(0).unsqueeze(1)
        return batch_mat


class CRX(ControlledRotationGate):
    """
    Primitive for the controlled RX gate.
    """

    def __init__(
        self,
        control: int | tuple[int, ...],
        target: int,
        param_name: str | int | float | Tensor = "",
        noise: DigitalNoiseProtocol | None = None,
    ):
        """Initializes controlled RX.

        Arguments:
            control: Control qubit(s).
            target: Target qubit.
            param_name: Name of parameters.
            noise: Optional noise protocols to apply.
        """
        super().__init__("X", control, target, param_name, noise=noise)

    @cached_property
    def eigenvals_generator(self) -> Tensor:
        """Get eigenvalues of the underlying generator.

        Arguments:
            values: Parameter values.

        Returns:
            Eigenvalues of the generator operator.
        """
        return torch.cat(
            (
                torch.zeros(
                    2 ** len(self.qubit_support) - 2,
                    device=self.device,
                    dtype=self.dtype,
                ),
                pauli_singleq_eigenvalues.flatten().to(
                    device=self.device, dtype=self.dtype
                ),
            )
        ).reshape(-1, 1)


class CRY(ControlledRotationGate):
    """
    Primitive for the controlled RY gate.
    """

    def __init__(
        self,
        control: int | tuple[int, ...],
        target: int,
        param_name: str | int | float | Tensor = "",
        noise: DigitalNoiseProtocol | None = None,
    ):
        """Initializes controlled RY.

        Arguments:
            control: Control qubit(s).
            target: Target qubit.
            param_name: Name of parameters.
            noise: Optional noise protocols to apply.
        """
        super().__init__("Y", control, target, param_name, noise=noise)

    @cached_property
    def eigenvals_generator(self) -> Tensor:
        """Get eigenvalues of the underlying generator.

        Arguments:
            values: Parameter values.

        Returns:
            Eigenvalues of the generator operator.
        """
        return torch.cat(
            (
                torch.zeros(
                    2 ** len(self.qubit_support) - 2,
                    device=self.device,
                    dtype=self.dtype,
                ),
                pauli_singleq_eigenvalues.flatten().to(
                    device=self.device, dtype=self.dtype
                ),
            )
        ).reshape(-1, 1)


class CRZ(ControlledRotationGate):
    """
    Primitive for the controlled RZ gate.
    """

    def __init__(
        self,
        control: int | tuple[int, ...],
        target: int,
        param_name: str | int | float | Tensor = "",
        noise: DigitalNoiseProtocol | None = None,
    ):
        """Initializes controlled RZ.

        Arguments:
            control: Control qubit(s).
            target: Target qubit.
            param_name: Name of parameters.
            noise: Optional noise protocols to apply.
        """
        super().__init__("Z", control, target, param_name, noise=noise)

    @cached_property
    def eigenvals_generator(self) -> Tensor:
        """Get eigenvalues of the underlying generator.

        Arguments:
            values: Parameter values.

        Returns:
            Eigenvalues of the generator operator.
        """
        return torch.cat(
            (
                torch.zeros(
                    2 ** len(self.qubit_support) - 2,
                    device=self.device,
                    dtype=self.dtype,
                ),
                pauli_singleq_eigenvalues.flatten().to(
                    device=self.device, dtype=self.dtype
                ),
            )
        ).reshape(-1, 1)


class CPHASE(ControlledRotationGate):
    """
    Primitive for the controlled PHASE gate.
    """

    n_params = 1

    def __init__(
        self,
        control: int | tuple[int, ...],
        target: int,
        param_name: str | int | float | Tensor = "",
        noise: DigitalNoiseProtocol | None = None,
    ):
        """Initializes controlled PHASE.

        Arguments:
            control: Control qubit(s).
            target: Target qubit.
            param_name: Name of parameters.
            noise: Optional noise protocols to apply.
        """
        super().__init__("I", control, target, param_name, noise=noise)

    @cached_property
    def eigenvals_generator(self) -> Tensor:
        """Get eigenvalues of the underlying generator.

        Arguments:
            values: Parameter values.

        Returns:
            Eigenvalues of the generator operator.
        """
        return torch.cat(
            (
                torch.tensor([-2.0, 0.0], device=self.device, dtype=self.dtype),
                torch.zeros(
                    2 ** len(self.qubit_support) - 2,
                    device=self.device,
                    dtype=self.dtype,
                ),
            )
        ).reshape(-1, 1)

    def _construct_parametric_base_op(
        self,
        values: dict[str, Tensor] | None = None,
        embedding: Embedding | None = None,
    ) -> Tensor:
        """
        Get the corresponding unitary.

        Arguments:
            values: A dict containing a Parameter name and value.
            embedding: An optional embedding for parameters.

        Returns:
            The unitary representation.
        """
        values = values or dict()
        thetas = self.parse_values(values, embedding)
        batch_size = len(thetas)
        mat = self.identity.unsqueeze(2).repeat(1, 1, batch_size)
        mat[1, 1, :] = torch.exp(1.0j * thetas).unsqueeze(0).unsqueeze(1)
        return controlled(mat, batch_size, len(self.control))

    def jacobian(
        self,
        values: dict[str, Tensor] | None = None,
        embedding: Embedding | None = None,
    ) -> Tensor:
        """
        Get the corresponding unitary of the jacobian.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation of the jacobian.
        """
        values = values or dict()
        thetas = self.parse_values(values, embedding)
        batch_size = len(thetas)
        n_control = len(self.control)
        jU = (
            torch.zeros((2, 2), dtype=self.identity.dtype)
            .unsqueeze(2)
            .repeat(1, 1, len(thetas))
        )
        jU[1, 1, :] = 1j * torch.exp(1j * thetas).unsqueeze(0).unsqueeze(1)
        n_dim = 2 ** (n_control + 1)
        jC = (
            torch.zeros((n_dim, n_dim), dtype=self.identity.dtype)
            .unsqueeze(2)
            .repeat(1, 1, batch_size)
        )
        unitary_idx = 2 ** (n_control + 1) - 2
        jC[unitary_idx:, unitary_idx:, :] = jU
        return jC

    def to(self, *args: Any, **kwargs: Any) -> Parametric:
        """Set device of primitive.

        Returns:
            Primitive with device.
        """
        super().to(*args, **kwargs)
        self._device = self.identity.device
        return self


class U(Parametric):
    r"""
    Primitive for the U gate.

    The corresponding unitary representation is:
    .. math::

        U = \begin{bmatrix}
                cos(\frac{\theta}{2}) & 4  \\ e^{-i \omega} sin(\frac{\theta}{2})
                e^{i \phi} sin(\frac{\theta}{2}) & e^{i (\phi+ \omega)} cos(\frac{\theta}{2})
            \end{bmatrix}

    where :math:`\phi`, :math:`\omega` and :math:`\theta` are input parameters.
    Attributes:
        phi: Phi parameter.
        theta: Theta parameter.
        omega: Omega parameter.
    """

    n_params = 3

    def __init__(
        self,
        target: int,
        phi: str,
        theta: str,
        omega: str,
        noise: DigitalNoiseProtocol | None = None,
    ):
        """Initializes U gate.

        Arguments:
            target: Target qubit.
            phi: Phi parameter.
            theta: Theta parameter.
            omega: Omega parameter.
            noise: Optional noise protocols to apply.
        """
        self.phi = phi
        self.theta = theta
        self.omega = omega
        super().__init__("X", target, param_name="", noise=noise)

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

    @cached_property
    def eigenvals_generator(self) -> Tensor:
        """Get eigenvalues of the underlying generator.

        Arguments:
            values: Parameter values.

        Returns:
            Eigenvalues of the generator operator.
        """
        return pauli_singleq_eigenvalues.to(device=self.device, dtype=self.dtype)

    def _construct_parametric_base_op(
        self,
        values: dict[str, Tensor] | None = None,
        embedding: Embedding | None = None,
    ) -> Tensor:
        """
        Get the corresponding unitary.

        Arguments:
            values: Parameter value.
            embedding: An optional embedding for parameters.

        Returns:
            The unitary representation.
        """
        if embedding is not None:
            raise NotImplementedError()
        values = values or dict()
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

    def jacobian(
        self,
        values: dict[str, Tensor] | None = None,
        embedding: Embedding | None = None,
    ) -> Tensor:
        """
        Get the corresponding unitary of the jacobian.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation of the jacobian.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def digital_decomposition(self) -> list[Parametric]:
        """
        Get the digital decomposition of U.

        Returns:
            The digital decomposition.
        """
        target = self.target[0]
        return [
            RZ(target, self.phi),
            RY(target, self.theta),
            RZ(target, self.omega),
        ]

    def jacobian_decomposed(
        self, values: dict[str, Tensor] | None = None
    ) -> list[Tensor]:
        """
        Get the corresponding unitary decomposition of the jacobian.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary decomposition of the jacobian.

        Raises:
            NotImplementedError
        """
        values = values or dict()
        return [op.jacobian(values) for op in self.digital_decomposition()]


OPS_PARAM_1Q = {PHASE, RX, RY, RZ, U}
OPS_PARAM_2Q = {CPHASE, CRX, CRY, CRZ}
OPS_PARAM = OPS_PARAM_1Q.union(OPS_PARAM_2Q)
OPS_DIAGONAL_PARAM = {RZ, PHASE, CRZ, CPHASE}
