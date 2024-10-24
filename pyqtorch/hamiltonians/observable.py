from __future__ import annotations

from math import log2

from torch import Tensor, trace, vmap
from torch.nn import Module, ParameterDict

from pyqtorch.apply import operator_product
from pyqtorch.circuit import Sequence
from pyqtorch.composite import Add
from pyqtorch.embed import Embedding
from pyqtorch.primitives import Primitive
from pyqtorch.utils import DensityMatrix, inner_prod


class Observable(Add):
    """
    The Observable :math:`O` represents an operator from which
    we can extract expectation values from quantum states.

    Given an input state :math:`\\ket\\rangle`, the expectation value with :math:`O` is defined as
    :math:`\\langle\\bra|O\\ket\\rangle`

    Attributes:
        operations: List of operations.
        n_qubits: Number of qubits it is defined on.
    """

    def __init__(
        self,
        operations: list[Module] | Primitive | Sequence,
    ):
        super().__init__(operations if isinstance(operations, list) else [operations])

    def expectation(
        self,
        state: Tensor,
        values: dict[str, Tensor] | ParameterDict | None = None,
        embedding: Embedding | None = None,
    ) -> Tensor:
        """Calculate the inner product :math:`\\langle\\bra|O\\ket\\rangle`

        Arguments:
            state: Input state.
            values: Values of parameters.

        Returns:
            The expectation value.
        """
        values = values or dict()
        if isinstance(state, DensityMatrix):
            n_qubits = int(log2(state.size()[0]))
            obs_rho = operator_product(
                self.tensor(values=values, embedding=embedding),
                self.qubit_support,
                state,
                tuple(range(n_qubits)),
            )
            return vmap(trace)(obs_rho.permute(2, 0, 1)).real

        return inner_prod(state, self.forward(state, values, embedding)).real
