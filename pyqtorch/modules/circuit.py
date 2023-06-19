from __future__ import annotations

from typing import Any

import torch
from torch.nn import Module, ModuleList, Parameter, init

from pyqtorch.modules.abstract import AbstractGate
from pyqtorch.modules.primitive import CNOT

PI = 2.0 * torch.asin(torch.Tensor([1.0]).double()).item()


def zero_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> torch.Tensor:
    """
    Generates the zero state for a specified number of qubits.

    Arguments:
        n_qubits (int): The number of qubits for which the zero state is to be generated.
        batch_size (int): The batch size for the zero state.
        device (str): The device on which the zero state tensor is to be allocated eg cpu or gpu.
        dtype (torch.cdouble): The data type of the zero state tensor.

    Returns:
        torch.Tensor: A tensor representing the zero state.
        The shape of the tensor is (batch_size, 2^n_qubits),
        where 2^n_qubits is the total number of possible states for the given number of qubits.
        The data type of the tensor is specified by the dtype parameter.

    Examples:
    ```python exec="on" source="above" result="json"
    import torch
    import pyqtorch.modules as pyq

    state = pyq.zero_state(n_qubits=2)
    print(state)  #tensor([[[1.+0.j],[0.+0.j]],[[0.+0.j],[0.+0.j]]], dtype=torch.complex128)
    ```
    """
    state = torch.zeros((2**n_qubits, batch_size), dtype=dtype, device=device)
    state[0] = 1
    state = state.reshape([2] * n_qubits + [batch_size])
    return state


def uniform_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> torch.Tensor:
    """
    Generates the uniform state for a specified number of qubits.
    Returns a tensor representing the uniform state.
    The shape of the tensor is (2^n_qubits, batch_size),
    where 2^n_qubits is the total number of possible states for the given number of qubits.
    The data type of the tensor is specified by the dtype parameter.
    Each element of the tensor is initialized to 1/sqrt(2^n_qubits),
    ensuring that the total probability of the state is equal to 1.

    Arguments:
        n_qubits (int): The number of qubits for which the uniform state is to be generated.
        batch_size (int): The batch size for the uniform state.
        device (str): The device on which the uniform state tensor is to be allocated.
        dtype (torch.cdouble): The data type of the uniform state tensor.

    Returns:
        torch.Tensor: A tensor representing the uniform state.


    Examples:
    ```python exec="on" source="above" result="json"
    import torch
    import pyqtorch.modules as pyq

    state = pyq.uniform_state(n_qubits=2)
    print(state)
    #tensor([[[0.5000+0.j],[0.5000+0.j]],[[0.5000+0.j],[0.5000+0.j]]], dtype=torch.complex128)
    ```
    """
    state = torch.ones((2**n_qubits, batch_size), dtype=dtype, device=device)
    state = state / torch.sqrt(torch.tensor(2**n_qubits))
    state = state.reshape([2] * n_qubits + [batch_size])
    return state


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
            return QuantumCircuit(
                max(self.n_qubits, other.n_qubits), self.operations.extend(other.operations)
            )

        if isinstance(other, AbstractGate):
            return QuantumCircuit(max(self.n_qubits, other.n_qubits), self.operations.append(other))

        else:
            return NotImplemented

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

    def is_same_circuit(self, other: QuantumCircuit) -> bool:
        return (
            all(
                gate1.is_same_gate(gate2) for gate1, gate2 in zip(self.operations, other.operations)
            )
            and self.n_qubits == other.n_qubits
        )


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
        for (op, t) in zip(self.operations, self.thetas):
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
