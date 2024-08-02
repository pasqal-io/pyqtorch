from __future__ import annotations

from typing import Callable

import pytest
import torch
from helpers import random_pauli_hamiltonian

import pyqtorch as pyq
from pyqtorch import DiffMode, expectation
from pyqtorch.analog import Observable
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.matrices import COMPLEX_TO_REAL_DTYPES
from pyqtorch.parametric import Parametric
from pyqtorch.utils import GPSR_ACCEPTANCE, PSR_ACCEPTANCE, GRADCHECK_sampling_ATOL


def circuit_psr(n_qubits: int) -> QuantumCircuit:
    """Helper function to make an example circuit using single gap PSR."""

    ops = [
        pyq.RX(0, "x"),
        pyq.RY(1, "y"),
        pyq.RX(0, "theta"),
        pyq.RY(1, torch.pi / 2),
        pyq.CNOT(0, 1),
    ]

    circ = QuantumCircuit(n_qubits, ops)

    return circ


def circuit_gpsr(n_qubits: int) -> QuantumCircuit:
    """Helper function to make an example circuit using multi gap GPSR."""

    ops = [
        pyq.Y(1),
        pyq.RX(0, "theta_0"),
        pyq.PHASE(0, "theta_1"),
        pyq.CSWAP(0, (1, 2)),
        pyq.CRX(1, 2, "theta_2"),
        pyq.CPHASE(1, 2, "theta_3"),
        pyq.CNOT(0, 1),
        pyq.Toffoli((0, 1), 2),
    ]

    circ = QuantumCircuit(n_qubits, ops)

    return circ


def circuit_sequence(n_qubits: int) -> QuantumCircuit:
    """Helper function to make an example circuit using Sequences of rotations."""
    name_angles = "theta"

    ops_rx = pyq.Sequence(
        [pyq.RX(i, param_name=name_angles + "_x_" + str(i)) for i in range(n_qubits)]
    )
    ops_rz = pyq.Sequence(
        [pyq.RZ(i, param_name=name_angles + "_z_" + str(i)) for i in range(n_qubits)]
    )
    cnot = pyq.CNOT(1, 2)
    ops = [ops_rx, ops_rz, cnot]
    circ = QuantumCircuit(n_qubits, ops)
    return circ


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
def test_expectation_gpsr(
    n_qubits: int,
    batch_size: int,
    circuit_fn: Callable,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(42)
    circ = circuit_fn(n_qubits).to(dtype)
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

    atol = PSR_ACCEPTANCE if circuit_fn != circuit_gpsr else GPSR_ACCEPTANCE

    # first order checks

    for i in range(len(grad_ad)):
        assert torch.allclose(grad_ad[i], grad_gpsr[i], atol=atol)
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
            assert torch.allclose(gradgrad_ad[j], gradgrad_gpsr[j], atol=atol)


@pytest.mark.parametrize("gate_type", ["scale", "hamevo", "same", ""])
@pytest.mark.parametrize("sequence_circuit", [True, False])
def test_compatibility_gpsr(gate_type: str, sequence_circuit: bool) -> None:

    pname = "theta_0"
    if gate_type == "scale":
        seq_gate = pyq.Sequence([pyq.X(0)])
        scale = pyq.Scale(seq_gate, pname)
        ops = [scale]
    elif gate_type == "hamevo":
        hamevo = pyq.HamiltonianEvolution(pyq.Sequence([pyq.X(0)]), pname, (0,))
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

    param_value = torch.pi / 2
    values = {"theta_0": torch.tensor([param_value], requires_grad=True)}

    if gate_type != "":
        with pytest.raises(ValueError):
            exp_gpsr = expectation(circ, state, values, obs, DiffMode.GPSR)

            grad_gpsr = torch.autograd.grad(
                exp_gpsr, tuple(values.values()), torch.ones_like(exp_gpsr)
            )
    else:
        exp_gpsr = expectation(circ, state, values, obs, DiffMode.GPSR)

        grad_gpsr = torch.autograd.grad(
            exp_gpsr, tuple(values.values()), torch.ones_like(exp_gpsr)
        )
        assert len(grad_gpsr) > 0
