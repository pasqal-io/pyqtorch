from qiskit import QuantumCircuit as QiskitCircuit
from qiskit import QuantumRegister
import torch

from pyqtorch.core.circuit import QuantumCircuit
from pyqtorch.converters.store_ops import ops_cache


# gate names mapping from PyQ to Qiskit
gates_map = {
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "cnot": "cx",
    "batchedrx": "rx",
    "batchedry": "ry",
    "batchedrz": "rz",
    "rzz": "rzz",
    "batchedrxx": "rxx",
    "batchedryy": "ryy",
    "batchedrzz": "rzz",
    "x": "x",
    "y": "y",
    "z": "z",
    "h": "h",
}


def pyq2qiskit(circuit: QuantumCircuit, *args, **kwargs) -> QiskitCircuit:
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
    _ = circuit(*args, **kwargs)
    assert len(ops_cache.operations) > 0, "Converting to Qiskit an empty circuit"

    # build the Qiskit circuit starting from the list of operations
    # in the PyQ circuit forward pass
    nqubits = circuit.n_qubits
    qr = QuantumRegister(nqubits, "q")
    qiskit_circuit = QiskitCircuit(qr)

    for op in ops_cache.operations:

        gate_name = gates_map[op.name]

        if op.param is not None:
            getattr(qiskit_circuit, gate_name)(op.param, *op.targets)
        else:
            getattr(qiskit_circuit, gate_name)(*op.targets)

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
