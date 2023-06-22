from __future__ import annotations

import torch

from pyqtorch import QuantumCircuit
from pyqtorch.core.batched_operation import batchedCPHASE, batchedCRX, batchedCRY, batchedCRZ

state_0 = torch.tensor([[1, 0]], dtype=torch.cdouble)
state_1 = torch.tensor([[0, 1]], dtype=torch.cdouble)

state_00 = torch.tensor([[1, 0], [0, 0]], dtype=torch.cdouble).unsqueeze(2)
state_10 = torch.tensor([[0, 1], [0, 0]], dtype=torch.cdouble).unsqueeze(2)
state_01 = torch.tensor([[0, 0], [1, 0]], dtype=torch.cdouble).unsqueeze(2)
state_11 = torch.tensor([[0, 0], [0, 1]], dtype=torch.cdouble).unsqueeze(2)

pi = torch.tensor(torch.pi, dtype=torch.cdouble)


def test_batched_ops() -> None:
    n_qubits: int = 2
    batch_size: int = 10
    qc = QuantumCircuit(n_qubits)

    theta_dim = torch.Size([batch_size])

    theta = torch.randn(theta_dim)
    psi = qc.uniform_state(batch_size)

    for op in [batchedCPHASE, batchedCRX, batchedCRY, batchedCRZ, batchedCPHASE]:
        res = op(theta, psi, [i for i in range(n_qubits)], n_qubits)
        assert not torch.any(torch.isnan(res))


def test_batched_cphase() -> None:
    n_qubits: int = 2
    psi = torch.tensor([[0, 0], [0, 1]], dtype=torch.cdouble).unsqueeze(2)
    psi_target = torch.tensor([[0, 0], [0, -1]], dtype=torch.cdouble).unsqueeze(2)
    angle = pi.unsqueeze(0)
    res = batchedCPHASE(angle, psi, [i for i in range(n_qubits)], n_qubits)
    assert torch.allclose(res, psi_target, atol=1e-16)
