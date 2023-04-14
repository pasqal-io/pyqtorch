from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit as QiskitCircuit
from qiskit import QuantumRegister

from pyqtorch.converters.store_ops import ops_cache
from pyqtorch.core.circuit import QuantumCircuit

# gate names mapping from PyQ to Qiskit
gates_map = {
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "CNOT": "cx",
    "RXX": "rxx",
    "RYY": "ryy",
    "RZZ": "rzz",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "H": "h",
    "U": "rv",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
    "SWAP": "swap",
    "CPHASE": "cp",
    "S": "s",
    "T": "t",
}


def pyq2qiskit(circuit: QuantumCircuit, *args: Any, **kwargs: Any) -> QiskitCircuit:
    """Convert a PyQ module into an equivalent circuit built with Qiskit

    This routine can be used for visualizing the quantum circuit models
    created with PyQ. Notice that in order to get the list of operations
    it needs to call a forward pass of the input QuantumCircuit instance0

    Args:
        circuit (QuantumCircuit): The PyQ QuantumCircuit instance to convert
            into Qiskit
        args, kwargs: Additional positional and keyworded arguments to pass to the
            circuit forward pass

    Returns:
        qiskit.QuantumCircuit: A QuantumCircuit instance from the Qiskit library
    """
    # execute the forward pass to populate the operation cache
    circuit.enable_converters()
    _ = circuit(*args, **kwargs)
    assert len(ops_cache.operations) > 0, "Converting to Qiskit an empty circuit"

    # build the Qiskit circuit starting from the list of operations
    # in the PyQ circuit forward pass
    nqubits = circuit.n_qubits
    qr = QuantumRegister(nqubits, "q")
    qiskit_circuit = QiskitCircuit(qr)

    for op in ops_cache.operations:
        gate_name = gates_map[op.name]

        if not hasattr(qiskit_circuit, gate_name):
            print(f"The gate {gate_name} is not available in Qiskit, conversion will be incomplete")
            continue

        if op.param is not None:
            if type(op.param) == float:
                op.param = [op.param]
            getattr(qiskit_circuit, gate_name)(*op.param, *op.targets)
        else:
            getattr(qiskit_circuit, gate_name)(*op.targets)

    circuit.disable_converters()

    return qiskit_circuit


# TODO: Implement the other way round
def qiskit2pyq(circuit: QiskitCircuit) -> QuantumCircuit:
    """Convert a circuit built with Qiskit into an equivalent PyQ module

    Args:
        circuit (QiskitCircuit): The input circuit to convert

    Returns:
        QuantumCircuit: The output PyQ circuit equivalent to the Qiskit input one
    """
    raise NotImplementedError
