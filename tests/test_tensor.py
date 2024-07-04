from __future__ import annotations

import random

import pytest
import torch
from conftest import _calc_mat_vec_wavefunction

from pyqtorch.primitive import OPS_1Q, OPS_2Q, OPS_3Q, OPS_DIGITAL, Toffoli
from pyqtorch.utils import (
    ATOL,
    RTOL,
    random_state,
)

pi = torch.tensor(torch.pi)


@pytest.mark.xfail
@pytest.mark.parametrize("n_qubits", [4, 5])
@pytest.mark.parametrize("batch_size", [1])
def test_digital_tensor(n_qubits: int, batch_size: int) -> None:
    for op in OPS_DIGITAL:
        if op in OPS_1Q:
            i = random.randint(0, n_qubits - 1)
            op_concrete = op(i)
        elif op in OPS_2Q:
            i, j = 0, random.randint(1, n_qubits - 1)
            op_concrete = op(i, j)
        elif op in OPS_3Q:
            i, j, k = 0, 1, random.randint(2, n_qubits - 1)
            op_concrete = op((i, j), k) if op == Toffoli else op(i, (j, k))

        psi_init = random_state(n_qubits, batch_size)
        psi_star = op_concrete(psi_init)
        psi_expected = _calc_mat_vec_wavefunction(
            op_concrete, n_qubits, psi_init, batch_size
        )
        assert torch.allclose(psi_star, psi_expected, rtol=RTOL, atol=ATOL)


# def test_hevo_constant_gen() -> None:
#     sup = (0, 1)
#     generator = pyq.Add(
#         [pyq.Scale(pyq.Z(0), torch.rand(1)), pyq.Scale(pyq.Z(1), torch.rand(1))]
#     )
#     hamevo = pyq.HamiltonianEvolution(generator, torch.rand(1), sup)
#     assert hamevo.generator_type == GeneratorType.OPERATION
#     psi = pyq.zero_state(2)
#     psi_star = hamevo(psi)
#     psi_expected = _calc_mat_vec_wavefunction(hamevo, len(sup), psi)
#     assert torch.allclose(psi_star, psi_expected, rtol=RTOL, atol=ATOL)


# @pytest.mark.parametrize("n_qubits", [2, 4, 6])
# @pytest.mark.parametrize("dim", [1, 2, 3, 4, 5, 6])
# @pytest.mark.parametrize("same_qubit_case", [True, False])
# def test_hamevo_tensor_from_circuit(
#     n_qubits: int, dim: int, same_qubit_case: bool
# ) -> None:
#     dim = min(n_qubits, dim)
#     vparam = "theta"
#     parametric = True
#     ops = [pyq.X, pyq.Y] * 2
#     if same_qubit_case:
#         qubit_targets = [dim] * len(ops)
#     else:
#         qubit_targets = np.random.choice(dim, len(ops), replace=True)
#     generator = pyq.QuantumCircuit(
#         n_qubits,
#         [
#             pyq.Add([op(q) for op, q in zip(ops, qubit_targets)]),
#             *[op(q) for op, q in zip(ops, qubit_targets)],
#         ],
#     )
#     generator = generator.tensor(n_qubits=n_qubits)
#     generator = generator + torch.conj(torch.transpose(generator, 0, 1))
#     hamevo = pyq.HamiltonianEvolution(
#         generator, vparam, tuple(range(n_qubits)), parametric
#     )
#     assert hamevo.generator_type == GeneratorType.TENSOR
#     vals = {vparam: torch.tensor([0.5])}
#     psi = random_state(n_qubits)
#     psi_star = hamevo(psi, vals)
#     psi_expected = _calc_mat_vec_wavefunction(hamevo, n_qubits, psi, vals)
#     assert torch.allclose(psi_star, psi_expected, rtol=RTOL, atol=ATOL)


# @pytest.mark.parametrize("n_qubits", [2, 4, 6])
# @pytest.mark.parametrize("dim", [1, 2, 3, 4, 5, 6])
# @pytest.mark.parametrize("same_qubit_case", [True, False])
# def test_hamevo_tensor_from_paramcircuit(
#     n_qubits: int, dim: int, same_qubit_case: bool
# ) -> None:
#     dim = min(n_qubits, dim)
#     tparam = "theta"
#     vparam = "rtheta"
#     parametric = True
#     ops = [pyq.RX, pyq.RY] * 2
#     if same_qubit_case:
#         qubit_targets = [dim] * len(ops)
#     else:
#         qubit_targets = np.random.choice(dim, len(ops), replace=True)
#     generator = pyq.QuantumCircuit(
#         n_qubits,
#         [
#             pyq.Add([op(q, vparam) for op, q in zip(ops, qubit_targets)]),
#             *[op(q, vparam) for op, q in zip(ops, qubit_targets)],
#         ],
#     )
#     generator = generator.tensor(
#         n_qubits=n_qubits, values={vparam: torch.tensor([0.5])}
#     )
#     generator = generator + torch.conj(torch.transpose(generator, 0, 1))
#     hamevo = pyq.HamiltonianEvolution(
#         generator, tparam, tuple(range(n_qubits)), parametric
#     )
#     assert hamevo.generator_type == GeneratorType.TENSOR
#     vals = {tparam: torch.tensor([0.5])}
#     psi = random_state(n_qubits)
#     psi_star = hamevo(psi, vals)
#     psi_expected = _calc_mat_vec_wavefunction(hamevo, n_qubits, psi, vals)
#     assert torch.allclose(psi_star, psi_expected, rtol=RTOL, atol=ATOL)
