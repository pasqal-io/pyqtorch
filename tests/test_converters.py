from qiskit import QuantumCircuit as QiskitCircuit

from pyqtorch.core.circuit import QuantumCircuit
from pyqtorch.converters.to_qiskit import pyq2qiskit, gates_map as qiskit_gates_map
from pyqtorch.converters.store_ops import ops_cache


def test_pyq2qiskit(test_circuit: QuantumCircuit):
    qiskit_circuit = pyq2qiskit(test_circuit)
    assert isinstance(qiskit_circuit, QiskitCircuit)
    assert qiskit_circuit.num_qubits == test_circuit.n_qubits
    assert len(ops_cache.operations) == len(qiskit_circuit.data)

    for pyq_op, qiskit_op in zip(ops_cache.operations, qiskit_circuit.data):
        assert qiskit_gates_map[pyq_op.name] == qiskit_op.operation.name
        # TODO: Add also additional checks for target qubits and parameters
