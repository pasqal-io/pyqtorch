from __future__ import annotations

from typing import Callable

import pytest
import torch
from helpers import random_parameter_names, random_pauli_hamiltonian

import pyqtorch as pyq
from pyqtorch import DiffMode, expectation
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.hamiltonians import HamiltonianEvolution, Observable
from pyqtorch.matrices import COMPLEX_TO_REAL_DTYPES, DEFAULT_MATRIX_DTYPE
from pyqtorch.primitives import Parametric
from pyqtorch.utils import PSR_ACCEPTANCE, GRADCHECK_sampling_ATOL


def Hamiltonian_general(n_qubits: int = 2, batch_size: int = 1) -> torch.Tensor:
    H_batch = torch.zeros(
        (2**n_qubits, 2**n_qubits, batch_size), dtype=DEFAULT_MATRIX_DTYPE
    )
    for i in range(batch_size):
        H_0 = torch.randn((2**n_qubits, 2**n_qubits), dtype=DEFAULT_MATRIX_DTYPE)
        H = (H_0 + torch.conj(H_0.transpose(0, 1))).to(DEFAULT_MATRIX_DTYPE)
        H_batch[..., i] = H
    return H_batch


def circuit_psr(n_qubits: int, different_names: bool = True) -> QuantumCircuit:
    """Helper function to make an example circuit using single gap PSR."""

    param_names = ["x", "y", "theta"]
    if not different_names:
        param_names = random_parameter_names(param_names)

    ops = [
        pyq.RX(0, param_names[0]),
        pyq.RY(1, param_names[1]),
        pyq.RX(0, param_names[2]),
        pyq.RY(1, torch.pi / 2),
        pyq.CNOT(0, 1),
    ]

    circ = QuantumCircuit(n_qubits, ops)

    return circ


def circuit_gpsr(n_qubits: int, different_names: bool = True) -> QuantumCircuit:
    """Helper function to make an example circuit using multi gap GPSR."""
    param_names = ["theta" + str(i) for i in range(4)]
    if not different_names:
        param_names = random_parameter_names(param_names)

    ops = [
        pyq.Y(1),
        pyq.RX(0, param_names[0]),
        pyq.PHASE(0, param_names[1]),
        pyq.CSWAP(0, (1, 2)),
        pyq.CRX(1, 2, param_names[2]),
        pyq.CPHASE(1, 2, param_names[3]),
        pyq.CNOT(0, 1),
        pyq.Toffoli((0, 1), 2),
    ]

    circ = QuantumCircuit(n_qubits, ops)

    return circ


def circuit_sequence(n_qubits: int, different_names: bool = True) -> QuantumCircuit:
    """Helper function to make an example circuit using Sequences of rotations."""
    name_angles = "theta"

    ops_rx = pyq.Sequence(
        [pyq.RX(i, param_name=name_angles + "_x_" + str(i)) for i in range(n_qubits)]
    )
    ops_rz = pyq.Sequence(
        [
            pyq.RZ(
                i,
                param_name=(
                    name_angles + "_z_" + str(i)
                    if different_names
                    else name_angles + "_z"
                ),
            )
            for i in range(n_qubits)
        ]
    )
    cnot = pyq.CNOT(1, 2)
    ops = [ops_rx, ops_rz, cnot]
    circ = QuantumCircuit(n_qubits, ops)
    return circ


def circuit_hamevo_tensor_gpsr(
    n_qubits: int, different_names: bool = True
) -> QuantumCircuit:
    """Helper function to make an example circuit."""

    ham = Hamiltonian_general(n_qubits)
    ham_op = pyq.HamiltonianEvolution(ham, "t", qubit_support=tuple(range(n_qubits)))
    param_names = ["theta" + str(i) for i in range(4)]
    if not different_names:
        param_names = random_parameter_names(param_names)

    ops = [
        pyq.CRX(0, 1, param_names[0]),
        pyq.X(1),
        pyq.CRY(1, 2, param_names[1]),
        ham_op,
        pyq.CRX(1, 2, param_names[2]),
        pyq.X(0),
        pyq.CRY(0, 1, param_names[3]),
        pyq.CNOT(0, 1),
    ]

    circ = QuantumCircuit(n_qubits, ops)

    return circ


def circuit_hamevo_pauligen_gpsr(
    n_qubits: int, different_names: bool = True
) -> QuantumCircuit:
    """Helper function to make an example circuit."""

    ham = random_pauli_hamiltonian(
        n_qubits, k_1q=n_qubits, k_2q=0, default_scale_coeffs=1.0
    )[0]
    ham_op = pyq.HamiltonianEvolution(ham, "t", qubit_support=tuple(range(n_qubits)))

    param_names = ["theta" + str(i) for i in range(4)]
    if not different_names:
        param_names = random_parameter_names(param_names)

    ops = [
        pyq.CRX(0, 1, param_names[0]),
        pyq.X(1),
        pyq.CRY(1, 2, param_names[1]),
        ham_op,
        pyq.CRX(1, 2, param_names[2]),
        pyq.X(0),
        pyq.CRY(0, 1, param_names[3]),
        pyq.CNOT(0, 1),
    ]

    circ = QuantumCircuit(n_qubits, ops)

    return circ


@pytest.mark.parametrize(
    ["n_qubits", "batch_size", "circuit_fn"],
    [
        (3, 1, circuit_hamevo_tensor_gpsr),
        (3, 1, circuit_hamevo_pauligen_gpsr),
    ],
)
@pytest.mark.parametrize("different_names", [False, True])
def test_expectation_gpsr_hamevo(
    n_qubits: int,
    batch_size: int,
    circuit_fn: Callable,
    different_names: bool,
    dtype: torch.dtype = torch.complex128,
) -> None:
    torch.manual_seed(42)
    circ = circuit_fn(n_qubits, different_names).to(dtype)
    obs = Observable(
        random_pauli_hamiltonian(
            n_qubits, k_1q=n_qubits, k_2q=0, default_scale_coeffs=1.0
        )[0]
    ).to(dtype)
    values = {
        op.param_name: torch.rand(
            batch_size, requires_grad=True, dtype=COMPLEX_TO_REAL_DTYPES[dtype]
        )
        for op in circ.flatten()
        if isinstance(op, Parametric) and isinstance(op.param_name, str)
    }
    values.update(
        {
            op.time: torch.rand(
                batch_size, requires_grad=True, dtype=COMPLEX_TO_REAL_DTYPES[dtype]
            )
            for op in circ.operations
            if isinstance(op, HamiltonianEvolution) and isinstance(op.time, str)
        }
    )
    state = pyq.random_state(n_qubits, dtype=dtype)

    # Apply adjoint
    exp_ad = expectation(circ, state, values, obs, DiffMode.AD)
    grad_ad = torch.autograd.grad(
        exp_ad, tuple(values.values()), torch.ones_like(exp_ad), create_graph=True
    )

    # Apply PSR
    exp_gpsr = expectation(circ, state, values, obs, DiffMode.GPSR)
    grad_gpsr = torch.autograd.grad(
        exp_gpsr, tuple(values.values()), torch.ones_like(exp_gpsr), create_graph=True
    )

    # first order checks
    for i in range(len(grad_ad)):
        assert torch.allclose(grad_ad[i], grad_gpsr[i], atol=PSR_ACCEPTANCE)

    # second order checks
    for i in range(len(grad_ad)):
        gradgrad_ad = torch.autograd.grad(
            grad_ad[i],
            tuple(values.values()),
            torch.ones_like(grad_ad[i]),
            create_graph=True,
        )

        gradgrad_gpsr = torch.autograd.grad(
            grad_gpsr[i],
            tuple(values.values()),
            torch.ones_like(grad_gpsr[i]),
            create_graph=True,
        )

        assert len(gradgrad_ad) == len(gradgrad_gpsr)

        # check second order gradients
        for j in range(len(gradgrad_ad)):
            assert torch.allclose(gradgrad_ad[j], gradgrad_gpsr[j], atol=1.0e-2)


@pytest.mark.parametrize(
    ["n_qubits", "batch_size", "circuit_fn"],
    [
        (2, 1, circuit_psr),
        (5, 10, circuit_psr),
        (3, 1, circuit_gpsr),
        (5, 10, circuit_gpsr),
        (3, 1, circuit_sequence),
        (5, 10, circuit_sequence),
    ],
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("different_names", [False, True])
def test_expectation_gpsr(
    n_qubits: int,
    batch_size: int,
    circuit_fn: Callable,
    dtype: torch.dtype,
    different_names: bool,
) -> None:
    torch.manual_seed(42)
    circ = circuit_fn(n_qubits, different_names).to(dtype)
    obs = Observable(
        random_pauli_hamiltonian(
            n_qubits, k_1q=n_qubits, k_2q=0, default_scale_coeffs=1.0
        )[0]
    ).to(dtype)
    values = {
        op.param_name: torch.rand(
            batch_size, requires_grad=True, dtype=COMPLEX_TO_REAL_DTYPES[dtype]
        )
        for op in circ.flatten()
        if isinstance(op, Parametric) and isinstance(op.param_name, str)
    }
    state = pyq.random_state(n_qubits, dtype=dtype)

    # Apply adjoint
    exp_ad = expectation(circ, state, values, obs, DiffMode.AD)
    grad_ad = torch.autograd.grad(
        exp_ad, tuple(values.values()), torch.ones_like(exp_ad), create_graph=True
    )

    # Apply PSR
    exp_gpsr = expectation(circ, state, values, obs, DiffMode.GPSR)
    grad_gpsr = torch.autograd.grad(
        exp_gpsr, tuple(values.values()), torch.ones_like(exp_gpsr), create_graph=True
    )

    exp_gpsr_sampled = expectation(
        circ,
        state,
        values,
        obs,
        DiffMode.GPSR,
        n_shots=100000,
    )
    grad_gpsr_sampled = torch.autograd.grad(
        exp_gpsr_sampled,
        tuple(values.values()),
        torch.ones_like(exp_gpsr_sampled),
        create_graph=True,
    )
    assert torch.allclose(exp_gpsr, exp_gpsr_sampled, atol=1e-01)

    # first order checks
    for i in range(len(grad_ad)):
        assert torch.allclose(grad_ad[i], grad_gpsr[i], atol=PSR_ACCEPTANCE)
        assert torch.allclose(
            grad_gpsr[i], grad_gpsr_sampled[i], atol=GRADCHECK_sampling_ATOL
        )

    # second order checks
    for i in range(len(grad_ad)):
        gradgrad_ad = torch.autograd.grad(
            grad_ad[i],
            tuple(values.values()),
            torch.ones_like(grad_ad[i]),
            create_graph=True,
        )

        gradgrad_gpsr = torch.autograd.grad(
            grad_gpsr[i],
            tuple(values.values()),
            torch.ones_like(grad_gpsr[i]),
            create_graph=True,
        )

        assert len(gradgrad_ad) == len(gradgrad_gpsr)

        # check second order gradients
        for j in range(len(gradgrad_ad)):
            assert torch.allclose(gradgrad_ad[j], gradgrad_gpsr[j], atol=PSR_ACCEPTANCE)


@pytest.mark.parametrize("gate_type", ["scale", "hamevo", "same", ""])
@pytest.mark.parametrize("sequence_circuit", [True, False])
def test_compatibility_gpsr(gate_type: str, sequence_circuit: bool) -> None:

    pname = "theta_0"
    if gate_type == "scale":
        seq_gate = pyq.Sequence([pyq.X(0)])
        scale = pyq.Scale(seq_gate, pname)
        ops = [scale]
    elif gate_type == "hamevo":
        symbol = pname
        t_evo = torch.tensor([torch.pi / 4], dtype=DEFAULT_MATRIX_DTYPE)
        hamevo = pyq.HamiltonianEvolution(symbol, t_evo, (0,))
        ops = [hamevo]
    elif gate_type == "same":
        ops = [pyq.RY(0, pname), pyq.RZ(0, pname)]
    else:
        # check that CNOT is not tested on spectral gap call
        ops = [pyq.RY(0, pname), pyq.CNOT(0, 1)]

    if sequence_circuit:
        circ = pyq.QuantumCircuit(2, pyq.Sequence(ops))
    else:
        circ = pyq.QuantumCircuit(2, ops)
    obs = pyq.Observable([pyq.Z(0)])
    state = pyq.zero_state(2)

    if gate_type == "hamevo":
        H = pyq.X(0).tensor()
        H.requires_grad = True
        values = {"theta_0": H}
    else:
        param_value = torch.pi / 2
        values = {"theta_0": torch.tensor([param_value], requires_grad=True)}

    if gate_type not in ("", "same"):
        with pytest.raises(ValueError):
            exp_gpsr = expectation(circ, state, values, obs, DiffMode.GPSR)
    else:
        exp_gpsr = expectation(circ, state, values, obs, DiffMode.GPSR)

        grad_gpsr = torch.autograd.grad(
            exp_gpsr, tuple(values.values()), torch.ones_like(exp_gpsr)
        )
        assert len(grad_gpsr) == 1
