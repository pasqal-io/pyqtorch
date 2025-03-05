from __future__ import annotations

from functools import cached_property
from typing import Any, Tuple
from uuid import uuid4

import torch
from torch import Tensor

from pyqtorch.embed import ConcretizedCallable, Embedding
from pyqtorch.matrices import (
    OPERATIONS_DICT,
    _jacobian,
    controlled,
    parametric_unitary,
)
from pyqtorch.noise import DigitalNoiseProtocol, _repr_noise
from pyqtorch.quantum_operation import QuantumOperation, Support

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
        param_name: str | int | float | torch.Tensor | ConcretizedCallable = "",
        noise: DigitalNoiseProtocol | None = None,
        diagonal: bool = False,
    ):
        """Initializes Parametric.

        Arguments:
            generator: Generator to use.
            qubit_support: Qubits to act on.
            param_name: Name of parameters.
            noise: Optional noise protocols to apply.
            diagonal: Whether the tensor generator is diagonal.
        """

        generator_operation = (
            OPERATIONS_DICT[generator] if isinstance(generator, str) else generator
        )
        if not isinstance(param_name, (str, int, float, Tensor, ConcretizedCallable)):
            raise TypeError(
                "Only str, int, float, Tensor or ConcretizedCallable types \
                are supported for param_name"
            )
        self.param_name = param_name
        self._param_uuid = str(uuid4())

        def parse_values(
            values: dict[str, Tensor] | Tensor | None = None,
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
            values = values or dict()
            if embedding is not None:
                values = embedding(values)
            # note: GPSR trick when the same param_name is used in many operations
            if self._param_uuid in values.keys():
                return Parametric._expand_values(values[self._param_uuid])  # type: ignore[index]
            return Parametric._expand_values(values[self.param_name])  # type: ignore[index]

        def parse_tensor(
            values: dict[str, Tensor] | Tensor | None = None,
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
            values = values or dict()
            # self.param_name will be ""
            return Parametric._expand_values(values)

        def parse_constant(
            values: dict[str, Tensor] | Tensor | None = None,
            embedding: Embedding | None = None,
        ) -> Tensor:
            """Fix a the parameter of a Parametric Gate to a numeric constant
               if the user passed a numeric input for the `param_name`.

            Arguments:
                values: A dict containing param_name:torch.Tensor pairs
            Returns:
                A Torch Tensor with which to evaluate the Parametric Gate.
            """
            values = values or dict()
            # self.param_name will be a torch.Tensor
            return Parametric._expand_values(
                torch.tensor(self.param_name, device=self.device, dtype=self.dtype)
            )

        def parse_concretized_callable(
            values: dict[str, Tensor] | Tensor | None = None,
            embedding: Embedding | None = None,
        ) -> Tensor:
            """Evaluate ConcretizedCallable object with given values.

            Arguments:
                values: A dict containing param_name:torch.Tensor pairs
            Returns:
                A Torch Tensor with which to evaluate the Parametric Gate.
            """
            # self.param_name will be a ConcretizedCallable
            values = values or dict()
            return Parametric._expand_values(self.param_name(values))  # type: ignore [operator]

        if param_name == "":
            self.parse_values = parse_tensor
            self.param_name = self.param_name
        elif isinstance(param_name, str):
            self.parse_values = parse_values
        elif isinstance(param_name, (float, int, torch.Tensor)):
            self.parse_values = parse_constant
        elif isinstance(param_name, ConcretizedCallable):
            self.parse_values = parse_concretized_callable

        # Parametric is defined by generator operation and a function
        # The function will use parsed parameter values to compute the unitary
        super().__init__(
            generator_operation,
            qubit_support,
            operator_function=self._construct_parametric_base_op,
            noise=noise,
            diagonal=diagonal if isinstance(generator, Tensor) else False,
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
        values: dict[str, Tensor] | Tensor | None = None,
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
        values = values or dict()
        thetas = self.parse_values(values, embedding)
        batch_size = len(thetas)
        mat = parametric_unitary(thetas, self.operation, self.identity, batch_size)
        return mat

    def jacobian(
        self,
        values: dict[str, Tensor] | Tensor | None = None,
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
        noise: DigitalNoiseProtocol | None = None,
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
        self,
        values: dict[str, Tensor] | None = None,
        embedding: Embedding | None = None,
    ) -> Tensor:
        """
        Get the corresponding unitary with parsed values and kronned identities
        for control.

        Arguments:
            values: A dict containing a Parameter name and value.
            embedding: An optional embedding for parameters.

        Returns:
            The unitary representation.
        """
        values = values or dict()
        thetas = self.parse_values(values, embedding)
        batch_size = len(thetas)
        mat = parametric_unitary(thetas, self.operation, self.identity, batch_size)
        mat = controlled(mat, batch_size, len(self.control))
        return mat

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
        noise: DigitalNoiseProtocol | None = None,
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
