from __future__ import annotations

from typing import Any

import torch
from torch.nn import Module, ModuleList, Parameter, init

from pyqtorch.modules.abstract import AbstractGate
from pyqtorch.modules.primitive import CNOT
from pyqtorch.modules.utils import zero_state

PI = 2.0 * torch.asin(torch.Tensor([1.0]).double()).item()


class QuantumCircuit(Module):
    def __init__(self, n_qubits: int, operations: list):
        """
        Creates a QuantumCircuit that can be used to compose multiple gates
        from a list of operations.

        Arguments:
            n_qubits (int): The total number of qubits in the circuit.
            operations (list): A list of gate operations to be applied in the circuit.

        Example:
        ```python exec="on" source="above" result="json"
        import torch
        import pyqtorch.modules as pyq

        #create a circuit with 2 qubits than provide a list of operations .
        #in this example we apply a X gate followed by a CNOT gate.
        circ = pyq.QuantumCircuit(
                                    n_qubits=2,
                                    operations=[
                                        pyq.X([0], 2),
                                        pyq.CNOT([0,1], 2)
                                    ]
                                )
        #create a zero state
        z = pyq.zero_state(2)

        #apply the circuit and its list of operations onto the zero state
        result=circ(z)

        #print the result
        print(result) #tensor([[[0.+0.j],[0.+0.j]],[[0.+0.j],[1.+0.j]]], dtype=torch.complex128)
        ```
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.operations = torch.nn.ModuleList(operations)

    def __mul__(self, other: AbstractGate | QuantumCircuit) -> QuantumCircuit:
        if isinstance(other, QuantumCircuit):
            n_qubits = max(self.n_qubits, other.n_qubits)
            return QuantumCircuit(n_qubits, self.operations.extend(other.operations))

        if isinstance(other, AbstractGate):
            n_qubits = max(self.n_qubits, other.n_qubits)
            return QuantumCircuit(n_qubits, self.operations.append(other))

        else:
            return ValueError(f"Cannot compose {type(self)} with {type(other)}")

    def __key(self) -> tuple:
        return (self.n_qubits, *self.operations)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, QuantumCircuit):
            return self.__key() == other.__key()
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.__key())

    def forward(self, state: torch.Tensor, thetas: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the quantum circuit.

        Arguments:
            state (torch.Tensor): The input quantum state tensor.
            thetas (torch.Tensor): Optional tensor of parameters for the circuit operations.

        Returns:
            torch.Tensor: The output quantum state tensor after applying the circuit operations.

        """
        for op in self.operations:
            state = op(state, thetas)
        return state

    @property
    def _device(self) -> torch.device:
        try:
            (_, buffer) = next(self.named_buffers())
            return buffer.device
        except StopIteration:
            return torch.device("cpu")

    def init_state(self, batch_size: int) -> torch.Tensor:
        return zero_state(self.n_qubits, batch_size, device=self._device)


def FeaturemapLayer(n_qubits: int, Op: Any) -> QuantumCircuit:
    """
    Creates a feature map layer in a quantum neural network.
    The FeaturemapLayer is a convenience constructor for a QuantumCircuit
    which accepts an operation to put on every qubit.

    Arguments:
        n_qubits (int): The total number of qubits in the circuit.
        Op (Any): The quantum operation to be applied in the feature map layer.

    Returns:
        QuantumCircuit: The feature map layer represented as a QuantumCircuit.

    Example:
    ```python exec="on" source="above" result="json"
    import torch
    import pyqtorch.modules as pyq

    #create a FeaturemapLayer to apply the RX operation on all 3 Qubits
    circ = pyq.FeaturemapLayer(n_qubits=3, Op=pyq.RX)
    print(circ)

    states = pyq.zero_state(n_qubits=3, batch_size=4)
    inputs = torch.rand(4)

    # the same batch of inputs are passed to the operations
    circ(states, inputs).shape
    ```
    """
    operations = [Op([i], n_qubits) for i in range(n_qubits)]
    return QuantumCircuit(n_qubits, operations)


class VariationalLayer(QuantumCircuit):
    def __init__(self, n_qubits: int, Op: Any):
        """
        Represents a variational layer in a quantum neural network allowing you
        to create a trainable QuantumCircuit.
        If you want the angles of your circuit to be trainable you can use a VariationalLayer.
        The VariationalLayer ignores the second input (because it has trainable angle parameters).

        Arguments:
            n_qubits (int): The total number of qubits in the circuit.
            Op (Any): The quantum operation to be applied in the variational layer.


        Example:
        ```python exec="on" source="above" result="json"
        import torch
        import pyqtorch.modules as pyq
        #create a variational layer with 3 qubits and operation of RX as the second parameter
        circ = pyq.VariationalLayer(n_qubits=3, Op=pyq.RX)
        state = pyq.zero_state(3)
        this_argument_is_ignored = None
        result=circ(state, this_argument_is_ignored)
        print(result)
        ```
        """
        operations = ModuleList([Op([i], n_qubits) for i in range(n_qubits)])
        super().__init__(n_qubits, operations)

        self.thetas = Parameter(torch.empty(n_qubits, Op.n_params))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.uniform_(self.thetas, -2 * PI, 2 * PI)

    def forward(self, state: torch.Tensor, _: torch.Tensor = None) -> torch.Tensor:
        for op, t in zip(self.operations, self.thetas):
            state = op(state, t)
        return state


class EntanglingLayer(QuantumCircuit):
    def __init__(self, n_qubits: int):
        """
        Represents an entangling layer in a quantum neural network by entangling Qubits

        Args:
            n_qubits (int): The total number of qubits in the circuit.

        Example:
        ```python exec="on" source="above" result="json"
        from pyqtorch.modules.circuit import EntanglingLayer

        # Create an entangling layer with 4 qubits
        entangling_layer = EntanglingLayer(n_qubits=4)

        print(entangling_layer)
        ```
        """
        operations = ModuleList(
            [CNOT([i % n_qubits, (i + 1) % n_qubits], n_qubits) for i in range(n_qubits)]
        )
        super().__init__(n_qubits, operations)
