from __future__ import annotations

import math

import torch
from numpy.typing import ArrayLike

from pyqtorch.core.operation import _apply_gate, create_controlled_matrix_from_operation
from pyqtorch.core.utils import OPERATIONS_DICT
from pyqtorch.modules.abstract import AbstractGate


class PrimitiveGate(AbstractGate):
    def __init__(self, gate: str, qubits: ArrayLike, n_qubits: int):
        super().__init__(qubits, n_qubits)
        self.gate = gate
        self.register_buffer("matrix", OPERATIONS_DICT[gate])

    def matrices(self, _: torch.Tensor) -> torch.Tensor:
        return self.matrix

    def apply(self, matrix: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return _apply_gate(state, matrix, self.qubits, self.n_qubits)

    def forward(self, state: torch.Tensor, _: torch.Tensor = None) -> torch.Tensor:
        return self.apply(self.matrix, state)


class X(PrimitiveGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        """
        Represents an X gate (Pauli-X gate) in a quantum circuit.
        The X gate class creates a  X gate that performs a PI rotation around the X axis

        Arguments:
            qubits (ArrayLike): The qubits to which the X gate is applied.
            n_qubits (int): The total number of qubits in the circuit.

        Examples:
        ```python exec="on" source="above" result="json"
        import torch
        import pyqtorch.modules as pyq

        # Create an X gate
        x_gate = pyq.X(qubits=[0], n_qubits=1)

        # Create a zero state
        z_state = pyq.zero_state(n_qubits=1)

        # Apply the X gate to the zero state
        result = x_gate(z_state)
        print(result)
        ```
        """
        super().__init__("X", qubits, n_qubits)


class Y(PrimitiveGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        """
        Represents a Y gate (Pauli-Y gate) in a quantum circuit.
        The Y gate class creates a  Y gate that performs a PI rotation around the Y axis.


        Arguments:
            qubits (ArrayLike): The qubits to which the Y gate is applied.
            n_qubits (int): The total number of qubits in the circuit.

        Examples:
        ```python exec="on" source="above" result="json"
        import torch
        import pyqtorch.modules as pyq

        # Create a Y gate
        y_gate = pyq.Y(qubits=[0], n_qubits=1)

        # Create a zero state
        z_state = pyq.zero_state(n_qubits=1)

        # Apply the Y gate to the zero state
        result = y_gate(z_state)
        print(result)
        ```
        """
        super().__init__("Y", qubits, n_qubits)


class Z(PrimitiveGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        """
        Represents a Z gate (Pauli-Z gate) in a quantum circuit.
        The ZGate class creates a Z gate that performs a PI rotation around the Z axis.

        Arguments:
            qubits (ArrayLike): The qubits to which the Z gate is applied.
            n_qubits (int): The total number of qubits in the circuit.

        Examples:
        ```python exec="on" source="above" result="json"
        import torch
        import pyqtorch.modules as pyq

        # Create a Z gate
        z_gate = pyq.Z(qubits=[0], n_qubits=1)

        # Create a zero state
        z_state = pyq.zero_state(n_qubits=1)

        # Apply the Z gate to the zero state
        result = z_gate(z_state)

        print(result)
        ```
        """
        super().__init__("Z", qubits, n_qubits)


# FIXME: do we really have to apply a matrix here?
# can't we just return the identical state?
class I(PrimitiveGate):  # noqa: E742
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        """
        Represents an I gate (identity gate) in a quantum circuit.
        The I gate class creates a I gate, which has no effect on the state of a qubit.

        Arguments:
            qubits (ArrayLike): The qubits to which the I gate is applied.
            n_qubits (int): The total number of qubits in the circuit.

        Examples:
        ```python exec="on" source="above" result="json"
        import torch
        import pyqtorch.modules as pyq

        # Create an I gate
        i_gate = pyq.I(qubits=[0], n_qubits=1)

        # Create a zero state
        z_state = pyq.zero_state(n_qubits=1)

        # Apply the I gate to the zero state
        result = i_gate(z_state)
        print(result)
        ```
        """
        super().__init__("I", qubits, n_qubits)


class H(PrimitiveGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        """
        Represents an H gate (Hadamard gate) in a quantum circuit.
        The H Gate class creates a H gate. It performs a PI rotation
        around the X+Z axis changing the basis from |0⟩,|1⟩ to  |+⟩,|-⟩
        and from |+⟩,|-⟩ back to |0⟩,|1⟩  depending on the number of times
        the gate is applied

        Arguments:
            qubits (ArrayLike): The qubits to which the H gate is applied.
            n_qubits (int): The total number of qubits in the circuit.

        Examples:
        ```python exec="on" source="above" result="json"
        import torch
        import pyqtorch.modules as pyq

        # Create an H gate
        h_gate = pyq.H(qubits=[0], n_qubits=1)

        # Create a zero state
        z_state = pyq.zero_state(n_qubits=1)

        # Apply the H gate to the zero state
        result = h_gate(z_state)
        print(result)  # tensor([[0.7071+0.j],[0.7071+0.j]], dtype=torch.complex128)
        ```
        """
        super().__init__("H", qubits, n_qubits)


class T(PrimitiveGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        """
        Represents a T gate (PI/4 phase gate) in a quantum circuit.

        Arguments:
            qubits (ArrayLike): The qubits to which the T gate is applied.
            n_qubits (int): The total number of qubits in the circuit.

        Examples:
        ```python exec="on" source="above" result="json"
        import torch
        import pyqtorch.modules as pyq

        # Create a T gate
        t_gate = pyq.T(qubits=[0], n_qubits=1)

        # Create a zero state
        z_state = pyq.zero_state(n_qubits=1)

        # Apply the T gate to the zero state
        result = t_gate(z_state)
        print(result)
        ```
        """
        super().__init__("T", qubits, n_qubits)


class S(PrimitiveGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        """
        Represents an S gate (PI/2 phase gate) in a quantum circuit.

        Arguments:
            qubits (ArrayLike): The qubits to which the S gate is applied.
            n_qubits (int): The total number of qubits in the circuit.

        Examples:
        ```python exec="on" source="above" result="json"
        import torch
        import pyqtorch.modules as pyq

        # Create an S gate
        s_gate = pyq.S(qubits=[0], n_qubits=1)

        # Create a zero state
        z_state = pyq.zero_state(n_qubits=1)

        # Apply the S gate to the zero state
        result = s_gate(z_state)
        print(result)
        ```
        """
        super().__init__("S", qubits, n_qubits)


class Sdagger(PrimitiveGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        """
        Represents an Sdagger gate (-PI/2 phase gate) in a quantum circuit.

        Arguments:
            qubits (ArrayLike): The qubits to which the S gate is applied.
            n_qubits (int): The total number of qubits in the circuit.

        Examples:
        ```python exec="on" source="above" result="json"
        import torch
        import pyqtorch.modules as pyq

        # Create an Sdagger gate
        sdagger_gate = pyq.Sdagger(qubits=[0], n_qubits=1)

        # Create a zero state
        z_state = pyq.zero_state(n_qubits=1)

        # Apply the Sdagger gate to the zero state
        result = sdagger_gate(z_state)
        print(result)
        ```
        """
        super().__init__("SDAGGER", qubits, n_qubits)


class SWAP(PrimitiveGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        """
        Represents a SWAP gate in a quantum circuit.
        The SwapGate swaps the qubit states of two quantum wires.


        Arguments:
            qubits (ArrayLike): The qubits to which the SWAP gate is applied.
            n_qubits (int): The total number of qubits in the circuit.

        Examples:
        ```python exec="on" source="above" result="json"
        import torch
        import pyqtorch.modules as pyq

        # Create a SWAP gate
        swap_gate = pyq.SWAP(qubits=[0, 1], n_qubits=2)

        # Create a zero state
        z_state = pyq.zero_state(n_qubits=2)

        # Apply the SWAP gate to the zero state
        result = swap_gate(z_state)
        print(result)
        ```
        """
        super().__init__("SWAP", qubits, n_qubits)


class ControlledOperationGate(AbstractGate):
    def __init__(self, gate: str, qubits: ArrayLike, n_qubits: int):
        super().__init__(qubits, n_qubits)
        self.gate = gate
        mat = OPERATIONS_DICT[gate]
        self.register_buffer(
            "matrix",
            create_controlled_matrix_from_operation(
                mat, n_control_qubits=len(qubits) - (int)(math.log2(mat.shape[0]))
            ),
        )

    def matrices(self, _: torch.Tensor) -> torch.Tensor:
        return self.matrix

    def apply(self, matrix: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return _apply_gate(state, matrix, self.qubits, self.n_qubits)

    def forward(self, state: torch.Tensor, _: torch.Tensor = None) -> torch.Tensor:
        return self.apply(self.matrix, state)


class CNOT(ControlledOperationGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        """
        Represents a controlled NOT (CNOT) gate in a quantum circuit.

        Arguments:
            qubits (ArrayLike): The control and target qubits for the CNOT gate.
            n_qubits (int): The total number of qubits in the circuit.

        Examples:
        ```python exec="on" source="above" result="json"
        import torch
        import pyqtorch.modules as pyq

        # Create a CNOT gate
        cnot_gate = pyq.CNOT(qubits=[0, 1], n_qubits=2)

        # Create a zero state
        z_state = pyq.zero_state(n_qubits=2)

        # Apply the CNOT gate to the zero state
        result = cnot_gate(z_state)
        print(result)
        ```
        """
        super().__init__("X", qubits, n_qubits)


CX = CNOT


class CY(ControlledOperationGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        """
        Represents a controlled-Y (CY) gate in a quantum circuit.
        The CY Gate class creates a controlled Y gate, applying the Y gate
        according to the control qubit state.

        Arguments:
            qubits (ArrayLike): The control and target qubits for the CY gate.
            n_qubits (int): The total number of qubits in the circuit.

        Examples:
        ```python exec="on" source="above" result="json"
        import torch
        import pyqtorch.modules as pyq

        # Create a CY gate
        cy_gate = pyq.CY(qubits=[0, 1], n_qubits=2)

        # Create a zero state
        z_state = pyq.zero_state(n_qubits=2)

        # Apply the CY gate to the zero state
        result = cy_gate(z_state)
        print(result)
        ```
        """
        super().__init__("Y", qubits, n_qubits)


class CZ(ControlledOperationGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        """
        Represents a controlled-Z (CZ) gate in a quantum circuit.
        The CZ gate class creates a controlled Z gate, applying
        the Z gate according to the control qubit state.


        Arguments:
            qubits (ArrayLike): The control and target qubits for the CZ gate.
            n_qubits (int): The total number of qubits in the circuit.

        Examples:
        ```python exec="on" source="above" result="json"
        import torch
        import pyqtorch.modules as pyq

        # Create a CZ gate
        cz_gate = pyq.CZ(qubits=[0, 1], n_qubits=2)

        # Create a zero state
        z_state = pyq.zero_state(n_qubits=2)

        # Apply the CZ gate to the zero state
        result = cz_gate(z_state)
        print(result)
        ```
        """
        super().__init__("Z", qubits, n_qubits)


class CSWAP(ControlledOperationGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        """
        Represents a controlled-SWAP (CSWAP) gate in a quantum circuit.
        The CSWAP gate class creates a controlled SWAP gate, applying
        the SWAP gate according to the control qubit state.


        Arguments:
            qubits (ArrayLike): The control and targets qubits for the CSWAP gate.
            n_qubits (int): The total number of qubits in the circuit.

        Examples:
        ```python exec="on" source="above" result="json"
        import torch
        import pyqtorch.modules as pyq

        # Create a CSWAP gate
        cswap_gate = pyq.CSWAP(qubits=[0, 1, 2], n_qubits=3)

        # Create a zero state
        swap_state = pyq.zero_state(n_qubits=3)

        # Apply the CSWAP gate to the zero state
        result = cswap_gate(swap_state)
        print(result)
        """
        super().__init__("SWAP", qubits, n_qubits)


class Toffoli(ControlledOperationGate):
    def __init__(self, qubits: ArrayLike, n_qubits: int):
        """
        Represents a multi qubit controlled toffoli gate in a quantum circuit.
        This gate performs a NOT operation only if all the control qubits are in state 1.
        Arguments:
            qubits (ArrayLike): The first n-1 qubits will be considered as the control
                                qubits and the last one will be the target qubit of the
                                Toffoli gate.

        Examples:
        ```python exec="on" source="above" result="json"
        import torch
        import pyqtorch.modules as pyq

        # Create a Toffoli gate with 2 control qubits.
        toffoli_gate = pyq.Toffoli(qubits=[0, 1, 2], n_qubits=3)

        # Create a one state
        state_1111 = pyq.X(qubits=[0], n_qubits=3)(pyq.X(qubits=[1], n_qubits=3)
        (pyq.X(qubits=[2], n_qubits=3)(pyq.zero_state(n_qubits=3)))

        # Apply the Toffoli gate to the zero state
        result = toffoli_gate(toffoli_state)
        print(result)
        """
        super().__init__("X", qubits, n_qubits)
