from __future__ import annotations

from typing import Any

import torch
from torch.nn import Module, ModuleList, Parameter, init

from pyqtorch.modules.primitive import CNOT

PI = 2.0 * torch.asin(torch.Tensor([1.0]).double()).item()


def zero_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> torch.Tensor:
    """
        Generate a zero state tensor for the specified number of qubits.

        Args:
            n_qubits (int): The number of qubits.
            batch_size (int, optional): The size of the batch (default: 1).
            device (str or torch.device, optional): The device to store the tensor (default: 'cpu').
            dtype (torch.dtype, optional): The data type of the tensor (default: torch.cdouble).

        Returns:
            torch.Tensor: A tensor representing the zero state.

        Examples:
            ```python exec="on" source="above" result="json"
            import torch
            from pyqtorch.modules import zero_state

            state = zero_state(n_qubits=2)
            print(state)
            # Output:
            # tensor([[1.+0.j],
            #         [0.+0.j],
            #         [0.+0.j],
            #         [0.+0.j]], dtype=torch.complex128)
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
       Generate a uniform state tensor for the specified number of qubits.

       Args:
           n_qubits (int): The number of qubits.
           batch_size (int, optional): The size of the batch (default: 1).
           device (str or torch.device, optional): The device to store the tensor (default: 'cpu').
           dtype (torch.dtype, optional): The data type of the tensor (default: torch.cdouble).

       Returns:
           torch.Tensor: A tensor representing the uniform state.

       Examples:
           ```python exec="on" source="above" result="json"
           import torch
           from pyqtorch.modules import uniform_state

           state = uniform_state(n_qubits=2)
           print(state)
           # Output:
           # tensor([[0.5+0.j],
           #         [0.5+0.j],
           #         [0.5+0.j],
           #         [0.5+0.j]], dtype=torch.complex128)
           ```
       """
    state = torch.ones((2**n_qubits, batch_size), dtype=dtype, device=device)
    state = state / torch.sqrt(torch.tensor(2**n_qubits))
    state = state.reshape([2] * n_qubits + [batch_size])
    return state


class QuantumCircuit(Module):
    def __init__(self, n_qubits: int, operations: list):
        """
                Initialize a quantum circuit module.

                Args:
                    n_qubits (int): The number of qubits.
                    operations (list): A list of operations to be applied in the circuit.

                Examples:
                    ```python exec="on" source="above" result="json"
                    from torch.nn import Linear
                    from pyqtorch.modules import QuantumCircuit

                    n_qubits = 2
                    operations = [Linear(2**n_qubits, 2**n_qubits), Linear(2**n_qubits, 2**n_qubits)]
                    circuit = QuantumCircuit(n_qubits, operations)
                    ```
                """
        super().__init__()
        self.n_qubits = n_qubits
        self.operations = torch.nn.ModuleList(operations)

    def forward(self, state: torch.Tensor, thetas: torch.Tensor = None) -> torch.Tensor:
        """
                Apply the quantum circuit to the input state.

                Args:
                    state (torch.Tensor): The input state tensor.
                    thetas (torch.Tensor, optional): The parameter tensor (default: None).

                Returns:
                    torch.Tensor: The output state tensor.

                Examples:
                    ```python exec="on" source="above" result="json"
                    import torch
                    from pyqtorch.modules import QuantumCircuit

                    n_qubits = 2
                    operations = [torch.nn.Linear(2**n_qubits, 2**n_qubits), torch.nn.Linear(2**n_qubits, 2**n_qubits)]
                    circuit = QuantumCircuit(n_qubits, operations)

                    input_state = torch.zeros((2**n_qubits, 1))
                    output_state = circuit(input_state)
                    print(output_state)
                    ```
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
        """
                Initialize the circuit's state.

                Args:
                    batch_size (int): The size of the batch.

                Returns:
                    torch.Tensor: The initialized state tensor.

                Examples:
                    ```python exec="on" source="above" result="json"
                    import torch
                    from pyqtorch.modules import QuantumCircuit

                    n_qubits = 2
                    operations = [torch.nn.Linear(2**n_qubits, 2**n_qubits), torch.nn.Linear(2**n_qubits, 2**n_qubits)]
                    circuit = QuantumCircuit(n_qubits, operations)

                    batch_size = 4
                    init_state = circuit.init_state(batch_size)
                    print(init_state)
                    ```
                """
        return zero_state(self.n_qubits, batch_size, device=self._device)


def FeaturemapLayer(n_qubits: int, Op: Any) -> QuantumCircuit:
    """
       Create a feature map layer for the quantum circuit.

       Args:
           n_qubits (int): The number of qubits.
           Op (Any): The operation to be applied to each qubit.

       Returns:
           QuantumCircuit: The feature map layer circuit.

       Examples:
           ```python exec="on" source="above" result="json"
           import torch
           from torch.nn import Linear
           from pyqtorch.modules import FeaturemapLayer

           n_qubits = 2
           op = Linear(2**n_qubits, 2**n_qubits)
           feature_map = FeaturemapLayer(n_qubits, op)
           ```
       """
    operations = [Op([i], n_qubits) for i in range(n_qubits)]
    return QuantumCircuit(n_qubits, operations)


class VariationalLayer(QuantumCircuit):
    def __init__(self, n_qubits: int, Op: Any):
        """
                Initialize a variational layer for the quantum circuit.

                Args:
                    n_qubits (int): The number of qubits.
                    Op (Any): The operation to be applied to each qubit.

                Examples:
                    ```python exec="on" source="above" result="json"
                    import torch
                    from torch.nn import Linear
                    from pyqtorch.modules import VariationalLayer

                    n_qubits = 2
                    op = Linear(2**n_qubits, 2**n_qubits)
                    var_layer = VariationalLayer(n_qubits, op)
                    ```
                """
        operations = ModuleList([Op([i], n_qubits) for i in range(n_qubits)])
        super().__init__(n_qubits, operations)

        self.thetas = Parameter(torch.empty(n_qubits, Op.n_params))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
                Reset the parameters of the variational layer.

                Examples:
                    ```python exec="on" source="above" result="json"
                    import torch
                    from torch.nn import Linear
                    from pyqtorch.modules import VariationalLayer

                    n_qubits = 2
                    op = Linear(2**n_qubits, 2**n_qubits)
                    var_layer = VariationalLayer(n_qubits, op)

                    var_layer.reset_parameters()
                    ```
                """
        init.uniform_(self.thetas, -2 * PI, 2 * PI)

    def forward(self, state: torch.Tensor, _: torch.Tensor = None) -> torch.Tensor:
        """
                Apply the variational layer to the input state.

                Args:
                    state (torch.Tensor): The input state tensor.
                    _ (torch.Tensor, optional): Placeholder argument for compatibility (default: None).

                Returns:
                    torch.Tensor: The output state tensor.

                Examples:
                    ```python exec="on" source="above" result="json"
                    import torch
                    from torch.nn import Linear
                    from pyqtorch.modules import VariationalLayer

                    n_qubits = 2
                    op = Linear(2**n_qubits, 2**n_qubits)
                    var_layer = VariationalLayer(n_qubits, op)

                    input_state = torch.zeros((2**n_qubits, 1))
                    output_state = var_layer(input_state)
                    print(output_state)
                    ```
                """
        for (op, t) in zip(self.operations, self.thetas):
            state = op(state, t)
        return state


class EntanglingLayer(QuantumCircuit):
    def __init__(self, n_qubits: int):
        """
               Initialize an entangling layer for the quantum circuit.

               Args:
                   n_qubits (int): The number of qubits.

               Examples:
                   ```python exec="on" source="above" result="json"
                   from pyqtorch.modules import EntanglingLayer

                   n_qubits = 2
                   entangling_layer = EntanglingLayer(n_qubits)
                   ```
               """
        operations = ModuleList(
            [CNOT([i % n_qubits, (i + 1) % n_qubits], n_qubits) for i in range(n_qubits)]
        )
        super().__init__(n_qubits, operations)
