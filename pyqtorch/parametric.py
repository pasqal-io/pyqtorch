from __future__ import annotations

from functools import cached_property
from typing import Any, Tuple

import torch
from torch import Tensor

from pyqtorch.embed import Embedding
from pyqtorch.matrices import (
    DEFAULT_MATRIX_DTYPE,
    OPERATIONS_DICT,
    _jacobian,
    controlled,
    parametric_unitary,
)
from pyqtorch.noise import NoiseProtocol, _repr_noise
from pyqtorch.quantum_ops import QuantumOperation, Support
from pyqtorch.utils import Operator

pauli_singleq_eigenvalues = torch.tensor([[-1.0], [1.0]], dtype=torch.cdouble)


class Parametric(QuantumOperation):
    """
    QuantumOperation taking parameters as input.

    Attributes:
        param_name: Name of parameters.
        parse_values: Method defining how to handle the values dictionary input.
    """

    n_params = 1

    def __init__(
        self,
        generator: str | Tensor,
        qubit_support: int | tuple[int, ...] | Support,
        param_name: str | int | float | torch.Tensor = "",
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        """Initializes Parametric.

        Arguments:
            generator: Generator to use.
            qubit_support: Qubits to act on.
            param_name: Name of parameters.
            noise: Optional noise protocols to apply.
        """

        generator_operation = (
            OPERATIONS_DICT[generator] if isinstance(generator, str) else generator
        )
        self.param_name = param_name

        def parse_values(
            values: dict[str, Tensor] | Tensor = dict(),
            embedding: Embedding | None = None,
        ) -> Tensor:
            """The legacy way of using parametric gates:
               The Parametric gate received a string as a 'param_name' and performs a
               a lookup in the passed `values` dict for to retrieve the torch.Tensor passed
               under the key `param_name`.

            Arguments:
                values: A dict containing param_name:torch.Tensor pairs
                embedding: An optional embedding.
            Returns:
                A Torch Tensor denoting values for the `param_name`.
            """
            return Parametric._expand_values(values[self.param_name])  # type: ignore[index]

        def parse_tensor(
            values: dict[str, Tensor] | Tensor = dict(),
            embedding: Embedding | None = None,
        ) -> Tensor:
            """Functional version of the Parametric gate:
               In case the user did not pass a `param_name`,
               pyqtorch assumes `values` will be a torch.Tensor instead of a dict.

            Arguments:
                values: A dict containing param_name:torch.Tensor pairs
            Returns:
                A Torch Tensor with which to evaluate the Parametric Gate.
            """
            # self.param_name will be ""
            return Parametric._expand_values(values)

        def parse_constant(
            values: dict[str, Tensor] | Tensor = dict(),
            embedding: Embedding | None = None,
        ) -> Tensor:
            """Fix a the parameter of a Parametric Gate to a numeric constant
               if the user passed a numeric input for the `param_name`.

            Arguments:
                values: A dict containing param_name:torch.Tensor pairs
            Returns:
                A Torch Tensor with which to evaluate the Parametric Gate.
            """
            # self.param_name will be a torch.Tensor
            return Parametric._expand_values(
                torch.tensor(self.param_name, device=self.device, dtype=self.dtype)
            )

        if param_name == "":
            self.parse_values = parse_tensor
            self.param_name = self.param_name
        elif isinstance(param_name, str):
            self.parse_values = parse_values
        elif isinstance(param_name, (float, int, torch.Tensor)):
            self.parse_values = parse_constant

        # Parametric is defined by generator operation and a function
        # The function will use parsed parameter values to compute the unitary
        super().__init__(
            generator_operation,
            qubit_support,
            operator_function=self._construct_parametric_base_op,
            noise=noise,
        )
        self.register_buffer("identity", OPERATIONS_DICT["I"])

    def extra_repr(self) -> str:
        """String representation of the operation.

        Returns:
            String with information on operation.
        """
        return f"target: {self.qubit_support}, param: {self.param_name}" + _repr_noise(
            self.noise
        )

    def __hash__(self) -> int:
        """Hash qubit support and param_name

        Returns:
            Hash value
        """
        return hash(self.qubit_support) + hash(self.param_name)

    @staticmethod
    def _expand_values(values: Tensor) -> Tensor:
        """Expand values if necessary.

        Arguments:
            values: Values of parameters
        Returns:
            Values of parameters expanded.

        """
        return values.unsqueeze(0) if len(values.size()) == 0 else values

    def _construct_parametric_base_op(
        self,
        values: dict[str, Tensor] | Tensor = dict(),
        embedding: Embedding | None = None,
    ) -> Tensor:
        """
        Get the corresponding unitary with parsed values.

        Arguments:
            values: A dict containing a Parameter name and value.
            embedding: An optional embedding for parameters.

        Returns:
            The unitary representation.
        """
        thetas = self.parse_values(values, embedding)
        batch_size = len(thetas)
        mat = parametric_unitary(thetas, self.operation, self.identity, batch_size)
        return mat

    def jacobian(
        self,
        values: dict[str, Tensor] | Tensor = dict(),
        embedding: Embedding | None = None,
    ) -> Tensor:
        """
        Get the corresponding unitary of the jacobian.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation of the jacobian.
        """
        thetas = self.parse_values(values, embedding)
        batch_size = len(thetas)
        return _jacobian(thetas, self.operation, self.identity, batch_size)

    def to(self, *args: Any, **kwargs: Any) -> Parametric:
        super().to(*args, **kwargs)
        self._device = self.operation.device
        self.param_name = (
            self.param_name.to(*args, **kwargs)
            if isinstance(self.param_name, torch.Tensor)
            else self.param_name
        )
        self._dtype = self.operation.dtype
        return self


class ControlledParametric(Parametric):
    """
    Primitives for controlled parametric operations.

    Attributes:
        control: Control qubit(s).
    """

    def __init__(
        self,
        operation: str | Tensor,
        control: int | Tuple[int, ...],
        target: int | Tuple[int, ...],
        param_name: str | int | float | torch.Tensor = "",
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        """Initializes a ControlledParametric.

        Arguments:
            operation: Rotation gate.
            control: Control qubit(s).
            qubit_targets: Target qubit(s).
            param_name: Name of parameters.
        """
        support = Support(target, control)
        super().__init__(operation, support, param_name, noise=noise)

    def extra_repr(self) -> str:
        """String representation of the operation.

        Returns:
            String with information on operation.
        """
        return (
            f"control: {self.control}, target: {(self.target,)}, param: {self.param_name}"
            + _repr_noise(self.noise)
        )

    def _construct_parametric_base_op(
        self, values: dict[str, Tensor] = dict(), embedding: Embedding | None = None
    ) -> Operator:
        """
        Get the corresponding unitary with parsed values and kronned identities
        for control.

        Arguments:
            values: A dict containing a Parameter name and value.
            embedding: An optional embedding for parameters.

        Returns:
            The unitary representation.
        """
        thetas = self.parse_values(values, embedding)
        batch_size = len(thetas)
        mat = parametric_unitary(thetas, self.operation, self.identity, batch_size)
        mat = controlled(mat, batch_size, len(self.control))
        return mat

    def jacobian(
        self, values: dict[str, Tensor] = dict(), embedding: Embedding | None = None
    ) -> Operator:
        """
        Get the corresponding unitary of the jacobian.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation of the jacobian.
        """
        thetas = self.parse_values(values, embedding)
        batch_size = len(thetas)
        n_control = len(self.control)
        jU = _jacobian(thetas, self.operation, self.identity, batch_size)
        n_dim = 2 ** (n_control + 1)
        jC = (
            torch.zeros((n_dim, n_dim), dtype=self.identity.dtype)
            .unsqueeze(2)
            .repeat(1, 1, batch_size)
        )
        unitary_idx = 2 ** (n_control + 1) - 2
        jC[unitary_idx:, unitary_idx:, :] = jU
        return jC


class ControlledRotationGate(ControlledParametric):
    """
    Primitives for controlled rotation operations.
    """

    n_params = 1

    def __init__(
        self,
        operation: str | Tensor,
        control: int | Tuple[int, ...],
        target: int,
        param_name: str | int | float | torch.Tensor = "",
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
    ):
        """Initializes a ControlledRotationGate.

        Arguments:
            gate: Rotation gate.
            control: Control qubit(s).
            qubit_support: Target qubit.
            param_name: Name of parameters.
        """
        super().__init__(operation, control, target, param_name, noise=noise)

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
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
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
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
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
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
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
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
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
        self, values: dict[str, Tensor] = dict(), embedding: Embedding | None = None
    ) -> Tensor:
        """
        Get the corresponding unitary.

        Arguments:
            values: A dict containing a Parameter name and value.
            embedding: An optional embedding for parameters.

        Returns:
            The unitary representation.
        """
        thetas = self.parse_values(values, embedding)
        batch_size = len(thetas)
        batch_mat = self.identity.unsqueeze(2).repeat(1, 1, batch_size)
        batch_mat[1, 1, :] = torch.exp(1.0j * thetas).unsqueeze(0).unsqueeze(1)
        return batch_mat

    def jacobian(
        self, values: dict[str, Tensor] = dict(), embedding: Embedding | None = None
    ) -> Tensor:
        """
        Get the corresponding unitary of the jacobian.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation of the jacobian.
        """
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
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
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
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
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
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
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
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
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
        self, values: dict[str, Tensor] = dict(), embedding: Embedding | None = None
    ) -> Tensor:
        """
        Get the corresponding unitary.

        Arguments:
            values: A dict containing a Parameter name and value.
            embedding: An optional embedding for parameters.

        Returns:
            The unitary representation.
        """
        thetas = self.parse_values(values, embedding)
        batch_size = len(thetas)
        mat = self.identity.unsqueeze(2).repeat(1, 1, batch_size)
        mat[1, 1, :] = torch.exp(1.0j * thetas).unsqueeze(0).unsqueeze(1)
        return controlled(mat, batch_size, len(self.control))

    def jacobian(
        self, values: dict[str, Tensor] = dict(), embedding: Embedding | None = None
    ) -> Tensor:
        """
        Get the corresponding unitary of the jacobian.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation of the jacobian.
        """
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
        noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None,
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
        self, values: dict[str, Tensor] = dict(), embedding: Embedding | None = None
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
        self, values: dict[str, Tensor] = {}, embedding: Embedding | None = None
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

    def jacobian_decomposed(self, values: dict[str, Tensor] = dict()) -> list[Tensor]:
        """
        Get the corresponding unitary decomposition of the jacobian.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary decomposition of the jacobian.

        Raises:
            NotImplementedError
        """
        return [op.jacobian(values) for op in self.digital_decomposition()]


OPS_PARAM_1Q = {PHASE, RX, RY, RZ, U}
OPS_PARAM_2Q = {CPHASE, CRX, CRY, CRZ}
OPS_PARAM = OPS_PARAM_1Q.union(OPS_PARAM_2Q)
