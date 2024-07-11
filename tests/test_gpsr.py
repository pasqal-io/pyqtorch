from __future__ import annotations

from typing import Callable

import pytest
import torch

import pyqtorch as pyq
from pyqtorch import DiffMode, expectation
from pyqtorch.analog import Observable
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.parametric import Parametric
from pyqtorch.utils import GPSR_ACCEPTANCE, PSR_ACCEPTANCE


def circuit_psr(n_qubits: int) -> QuantumCircuit:
    """Helper function to make an example circuit."""

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
    """Helper function to make an example circuit."""

    ops = [
        pyq.Y(1),
        pyq.RX(0, "theta_0"),
        pyq.PHASE(0, "theta_1"),
        pyq.CSWAP(0, (1, 2)),
        pyq.CRX(1, 2, "theta_2"),
        pyq.CPHASE(1, 2, "theta_3"),
        pyq.CNOT(0, 1),
        pyq.Toffoli((2, 1), 0),
    ]

    circ = QuantumCircuit(n_qubits, ops)

    return circ

@pytest.mark.parametrize(
    ["n_qubits", "batch_size", "n_obs", "circuit_fn"],
    [
        (2, 1, 2, circuit_psr),
        (5, 10, 1, circuit_psr),
        (3, 1, 2, circuit_gpsr),
        (5, 10, 1, circuit_gpsr),
    ],
)
def test_expectation_psr(
    n_qubits: int, batch_size: int, n_obs: int, circuit_fn: Callable
) -> None:
    torch.manual_seed(42)
    circ = circuit_fn(n_qubits)
    obs = Observable(n_qubits, [pyq.Z(i) for i in range(n_qubits)])

    values = {
        op.param_name: torch.rand(batch_size, requires_grad=True)
        for op in circ.flatten()
        if isinstance(op, Parametric)
        and isinstance(op.param_name, str)
    }
    state = pyq.random_state(n_qubits)

    # Apply adjoint
    exp_ad = expectation(circ, state, values, obs, DiffMode.AD)
    grad_ad = torch.autograd.grad(
        exp_ad, tuple(values.values()), torch.ones_like(exp_ad), create_graph=True
    )[0]
    gradgrad_ad = torch.autograd.grad(
        grad_ad, tuple(values.values()), torch.ones_like(grad_ad)
    )

    # Apply PSR
    exp_gpsr = expectation(circ, state, values, obs, DiffMode.GPSR)
    grad_gpsr = torch.autograd.grad(
        exp_gpsr, tuple(values.values()), torch.ones_like(exp_gpsr), create_graph=True
    )[0]
    gradgrad_gpsr = torch.autograd.grad(
        grad_gpsr, tuple(values.values()), torch.ones_like(grad_gpsr)
    )

    atol = PSR_ACCEPTANCE if circuit_fn == circuit_psr else GPSR_ACCEPTANCE

    assert torch.allclose(exp_ad, exp_gpsr)

    for i in range(len(grad_ad)):
        assert torch.allclose(grad_ad[i], grad_gpsr[i], atol=atol)

    for i in range(len(gradgrad_ad)):
        assert torch.allclose(gradgrad_ad[i], gradgrad_gpsr[i], atol=atol)


@pytest.mark.parametrize("gate_type", ["scale", "hamevo", ""])
def test_compatibility_gpsr(gate_type: str) -> None:

    pname = "theta_0"
    if gate_type == "scale":
        seq_gate = pyq.Sequence([pyq.X(0)])
        scale = pyq.Scale(seq_gate, pname)
        ops = [scale]
    elif gate_type == "hamevo":
        hamevo = pyq.HamiltonianEvolution(pyq.Sequence([pyq.X(0)]), pname, (0,))
        ops = [hamevo]
    else:
        ops = [pyq.RY(0, pname), pyq.RZ(0, pname)]

    circ = pyq.QuantumCircuit(1, ops)
    obs = pyq.QuantumCircuit(1, [pyq.Z(0)])
    state = pyq.zero_state(1)

    param_value = torch.pi / 2
    values = {"theta_0": torch.tensor([param_value], requires_grad=True)}
    with pytest.raises(ValueError):
        exp_gpsr = expectation(circ, state, values, obs, DiffMode.GPSR)

        grad_gpsr = torch.autograd.grad(
            exp_gpsr, tuple(values.values()), torch.ones_like(exp_gpsr)
        )
