from __future__ import annotations

from qiskit import QuantumCircuit as QiskitCircuit

from pyqtorch.converters.store_ops import ops_cache
from pyqtorch.converters.to_qiskit import gates_map as qiskit_gates_map
from pyqtorch.converters.to_qiskit import pyq2qiskit
from pyqtorch.core.circuit import QuantumCircuit


def test_pyq2qiskit(test_circuit: QuantumCircuit) -> None:
    qiskit_circuit = pyq2qiskit(test_circuit)
    assert isinstance(qiskit_circuit, QiskitCircuit)
    assert qiskit_circuit.num_qubits == test_circuit.n_qubits
    assert len(ops_cache.operations) == len(qiskit_circuit.data)

    for pyq_op, qiskit_op in zip(ops_cache.operations, qiskit_circuit.data):
        assert qiskit_gates_map[pyq_op.name] == qiskit_op.operation.name
        # TODO: Add also additional checks for target qubits and parameters


if __name__ == "__main__":
    from pyqtorch.ansatz import AlternateLayerAnsatz

    n_qubits = 4
    n_layers = 2
    ansatz = AlternateLayerAnsatz(n_qubits, n_layers)
    state = QuantumCircuit(n_qubits).init_state(1)
    qiskit_ansatz = pyq2qiskit(ansatz, state)
    print(qiskit_ansatz)
