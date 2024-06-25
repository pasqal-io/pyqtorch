from __future__ import annotations

from typing import Any, Tuple

import torch
from torch import Tensor

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
    """
    Primitives taking parameters as input.

    Attributes:
        param_name: Name of parameters.
        param_values: Values of parameters provided at initialization.
        parse_values: Method defining how to handle the values dictionary input.
    """

    n_params = 1

    def __init__(
        self,
        generator_name: str,
        target: int,
        param_name: str = "",
        param_values: Tensor = None,
    ):
        """Initializes Parametric.

        Arguments:
            generator_name: Name of the operation.
            target: Target qubit.
            param_name: Name of parameters.
            param_values: Values of parameters provided at initialization.
        """
        super().__init__(OPERATIONS_DICT[generator_name], target)
        self.register_buffer("identity", OPERATIONS_DICT["I"])
        self.param_name = param_name
        self.param_value = param_values

        def parse_values(values: dict[str, Tensor] | Tensor = dict()) -> Tensor:
            """Get from values input dictionary the values for param_name.

            Arguments:
                values: Values of parameters
            Returns:
                Parameters values from values.
            """
            return Parametric._expand_values(values[self.param_name])

        def parse_tensor(values: dict[str, Tensor] | Tensor = dict()) -> Tensor:
            """Return values if param_name empty string.

            Arguments:
                values: Values of parameters
            Returns:
                Values of parameters.
            """
            return Parametric._expand_values(values)

        def parse_default_values(values: dict[str, Tensor] | Tensor = dict()) -> Tensor:
            """Return default parameter values.

            Arguments:
                values: Values of parameters
            Returns:
                Values of parameters.
            """
            return Parametric._expand_values(self.param_value)

        if param_values is not None:
            self.parse_values = parse_default_values
        else:
            self.parse_values = parse_tensor if param_name == "" else parse_values

    def extra_repr(self) -> str:
        """String representation of the operation.

        Returns:
            String with information on operation.
        """
        return f"target:{self.qubit_support}, param:{self.param_name}"

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

    def unitary(self, values: dict[str, Tensor] | Tensor = dict()) -> Operator:
        """
        Get the corresponding unitary.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation.
        """
        thetas = self.parse_values(values)
        batch_size = len(thetas)
        return _unitary(thetas, self.pauli, self.identity, batch_size)

    def jacobian(self, values: dict[str, Tensor] | Tensor = dict()) -> Operator:
        """
        Get the corresponding unitary of the jacobian.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation of the jacobian.
        """
        thetas = self.parse_values(values)
        batch_size = len(thetas)
        return _jacobian(thetas, self.pauli, self.identity, batch_size)


class RX(Parametric):
    """
    Primitive for the RX gate.

    The corresponding unitary is obtained by :math:`exp(-i X, t)`
    where :math:`t` is an input parameter
    and :math:`X` is the matrix representation of the X gate.

    Attributes:
        param_name: Name of parameters.
        param_values: Values of parameters provided at initialization.
        parse_values: Method defining how to handle the values dictionary input.
    """

    def __init__(
        self,
        target: int,
        param_name: str = "",
        param_values: Tensor = None,
    ):
        """Initializes RX.

        Arguments:
            target: Target qubit.
            param_name: Name of parameters.
            param_values: Values of parameters provided at initialization.
        """
        super().__init__("X", target, param_name, param_values)


class RY(Parametric):
    """
    Primitive for the RY gate.

    The corresponding unitary is obtained by :math:`exp(-i Y, t)`
    where :math:`t` is an input parameter
    and :math:`Y` is the matrix representation of the Y gate.

    Attributes:
        param_name: Name of parameters.
        param_values: Values of parameters provided at initialization.
        parse_values: Method defining how to handle the values dictionary input.
    """

    def __init__(
        self,
        target: int,
        param_name: str = "",
        param_values: Tensor = None,
    ):
        """Initializes RY.

        Arguments:
            target: Target qubit.
            param_name: Name of parameters.
            param_values: Values of parameters provided at initialization.
        """
        super().__init__("Y", target, param_name, param_values)


class RZ(Parametric):
    """
    Primitive for the RZ gate.

    The corresponding unitary is obtained by :math:`exp(-i Z, t)`
    where :math:`t` is an input parameter
    and :math:`Z` is the matrix representation of the Z gate.

    Attributes:
        param_name: Name of parameters.
        param_values: Values of parameters provided at initialization.
        parse_values: Method defining how to handle the values dictionary input.
    """

    def __init__(
        self,
        target: int,
        param_name: str = "",
        param_values: Tensor = None,
    ):
        """Initializes RZ.

        Arguments:
            target: Target qubit.
            param_name: Name of parameters.
            param_values: Values of parameters provided at initialization.
        """
        super().__init__("Z", target, param_name, param_values)


class PHASE(Parametric):
    """
    Primitive for the PHASE gate.

    The corresponding unitary is obtained by :math:`exp(-i I, t)`
    where :math:`t` is an input parameter
    and :math:`I` is the identity.

    Attributes:
        param_name: Name of parameters.
        param_values: Values of parameters provided at initialization.
        parse_values: Method defining how to handle the values dictionary input.
    """

    def __init__(
        self,
        target: int,
        param_name: str = "",
        param_values: Tensor = None,
    ):
        """Initializes PHASE.

        Arguments:
            target: Target qubit.
            param_name: Name of parameters.
            param_values: Values of parameters provided at initialization.
        """
        super().__init__("I", target, param_name, param_values)

    def unitary(self, values: dict[str, Tensor] = dict()) -> Operator:
        """
        Get the corresponding unitary.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation.
        """
        thetas = self.parse_values(values)
        batch_size = len(thetas)
        batch_mat = self.identity.unsqueeze(2).repeat(1, 1, batch_size)
        batch_mat[1, 1, :] = torch.exp(1.0j * thetas).unsqueeze(0).unsqueeze(1)
        return batch_mat

    def jacobian(self, values: dict[str, Tensor] = dict()) -> Operator:
        """
        Get the corresponding unitary of the jacobian.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation of the jacobian.
        """
        thetas = self.parse_values(values)
        batch_mat = (
            torch.zeros((2, 2), dtype=self.identity.dtype)
            .unsqueeze(2)
            .repeat(1, 1, len(thetas))
        )
        batch_mat[1, 1, :] = 1j * torch.exp(1j * thetas).unsqueeze(0).unsqueeze(1)
        return batch_mat


class ControlledRotationGate(Parametric):
    """
    Primitives for controlled rotation operations.

    Attributes:
        control: Control qubit(s).
        qubit_support: Qubits acted on.
    """

    n_params = 1

    def __init__(
        self,
        gate: str,
        control: int | Tuple[int, ...],
        target: int,
        param_name: str = "",
        param_values: Tensor = None,
    ):
        """Initializes a ControlledRotationGate.

        Arguments:
            gate: Rotation gate.
            control: Control qubit(s).
            target: Target qubit.
            param_name: Name of parameters.
            param_values: Values of parameters provided at initialization.
        """
        self.control = control if isinstance(control, tuple) else (control,)
        super().__init__(gate, target, param_name, param_values)
        self.qubit_support = self.control + (self.target,)  # type: ignore[operator]
        # In this class, target is always an int but herit from Parametric and Primitive that:
        # target : int | tuple[int,...]

    def extra_repr(self) -> str:
        """String representation of the operation.

        Returns:
            String with information on operation.
        """
        return (
            f"control: {self.control}, target:{(self.target,)}, param:{self.param_name}"
        )

    def unitary(self, values: dict[str, Tensor] = dict()) -> Operator:
        """
        Get the corresponding unitary.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation.
        """
        thetas = self.parse_values(values)
        batch_size = len(thetas)
        mat = _unitary(thetas, self.pauli, self.identity, batch_size)
        return _controlled(mat, batch_size, len(self.control))

    def jacobian(self, values: dict[str, Tensor] = dict()) -> Operator:
        """
        Get the corresponding unitary of the jacobian.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation of the jacobian.
        """
        thetas = self.parse_values(values)
        batch_size = len(thetas)
        n_control = len(self.control)
        jU = _jacobian(thetas, self.pauli, self.identity, batch_size)
        n_dim = 2 ** (n_control + 1)
        jC = (
            torch.zeros((n_dim, n_dim), dtype=self.identity.dtype)
            .unsqueeze(2)
            .repeat(1, 1, batch_size)
        )
        unitary_idx = 2 ** (n_control + 1) - 2
        jC[unitary_idx:, unitary_idx:, :] = jU
        return jC


class CRX(ControlledRotationGate):
    """
    Primitive for the controlled RX gate.
    """

    def __init__(
        self,
        control: int | Tuple[int, ...],
        target: int,
        param_name: str = "",
        param_values: Tensor = None,
    ):
        """Initializes controlled RX.

        Arguments:
            control: Control qubit(s).
            target: Target qubit.
            param_name: Name of parameters.
            param_values: Values of parameters provided at initialization.
        """
        super().__init__("X", control, target, param_name, param_values)


class CRY(ControlledRotationGate):
    """
    Primitive for the controlled RY gate.
    """

    def __init__(
        self,
        control: int | Tuple[int, ...],
        target: int,
        param_name: str = "",
        param_values: Tensor = None,
    ):
        """Initializes controlled RY.

        Arguments:
            control: Control qubit(s).
            target: Target qubit.
            param_name: Name of parameters.
            param_values: Values of parameters provided at initialization.
        """
        super().__init__("Y", control, target, param_name, param_values)


class CRZ(ControlledRotationGate):
    """
    Primitive for the controlled RZ gate.
    """

    def __init__(
        self,
        control: int | Tuple[int, ...],
        target: int,
        param_name: str = "",
        param_values: Tensor = None,
    ):
        """Initializes controlled RZ.

        Arguments:
            control: Control qubit(s).
            target: Target qubit.
            param_name: Name of parameters.
            param_values: Values of parameters provided at initialization.
        """
        super().__init__("Z", control, target, param_name, param_values)


class CPHASE(ControlledRotationGate):
    """
    Primitive for the controlled PHASE gate.
    """

    n_params = 1

    def __init__(
        self,
        control: int | Tuple[int, ...],
        target: int,
        param_name: str = "",
        param_values: Tensor = None,
    ):
        """Initializes controlled PHASE.

        Arguments:
            control: Control qubit(s).
            target: Target qubit.
            param_name: Name of parameters.
            param_values: Values of parameters provided at initialization.
        """
        super().__init__("I", control, target, param_name, param_values)

    def unitary(self, values: dict[str, Tensor] = dict()) -> Operator:
        """
        Get the corresponding unitary.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation.
        """
        thetas = self.parse_values(values)
        batch_size = len(thetas)
        mat = self.identity.unsqueeze(2).repeat(1, 1, batch_size)
        mat[1, 1, :] = torch.exp(1.0j * thetas).unsqueeze(0).unsqueeze(1)
        return _controlled(mat, batch_size, len(self.control))

    def jacobian(self, values: dict[str, Tensor] = dict()) -> Operator:
        """
        Get the corresponding unitary of the jacobian.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation of the jacobian.
        """
        thetas = self.parse_values(values)
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

    def to(self, *args: Any, **kwargs: Any) -> Primitive:
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

    def __init__(self, target: int, phi: str, theta: str, omega: str):
        """Initializes U gate.

        Arguments:
            phi: Phi parameter.
            theta: Theta parameter.
            omega: Omega parameter.
        """
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

    def unitary(self, values: dict[str, Tensor] = dict()) -> Operator:
        """
        Get the corresponding unitary.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation.
        """
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

    def jacobian(self, values: dict[str, Tensor] = {}) -> Operator:
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
        return [
            RZ(self.qubit_support[0], self.phi),
            RY(self.qubit_support[0], self.theta),
            RZ(self.qubit_support[0], self.omega),
        ]

    def jacobian_decomposed(self, values: dict[str, Tensor] = dict()) -> list[Operator]:
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
