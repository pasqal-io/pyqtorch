from __future__ import annotations

import random

import pytest
import torch
from conftest import _calc_mat_vec_wavefunction

from pyqtorch.parametric import OPS_PARAM, OPS_PARAM_1Q, OPS_PARAM_2Q
from pyqtorch.primitive import (
    OPS_1Q,
    OPS_2Q,
    OPS_3Q,
    OPS_DIGITAL,
    Toffoli,
)
from pyqtorch.utils import (
    ATOL,
    RTOL,
    random_state,
)

pi = torch.tensor(torch.pi)


@pytest.mark.parametrize("n_qubits", [4, 5])
@pytest.mark.parametrize("batch_size", [1, 3, 5])
def test_digital_tensor(n_qubits: int, batch_size: int) -> None:
    for op in OPS_DIGITAL:
        if op in OPS_1Q:
            supp: tuple = (random.randint(0, n_qubits - 1),)
        elif op in OPS_2Q:
            supp = (0, random.randint(1, n_qubits - 1))
        elif op in OPS_3Q:
            i, j, k = 0, 1, random.randint(2, n_qubits - 1)
            supp = ((i, j), k) if op == Toffoli else (i, (j, k))
        op_concrete = op(*supp)
        psi_init = random_state(n_qubits, batch_size)
        psi_star = op_concrete(psi_init)
        psi_expected = _calc_mat_vec_wavefunction(op_concrete, psi_init)
        assert torch.allclose(psi_star, psi_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("batch_size", [1, 3, 5])
@pytest.mark.parametrize("n_qubits", [4, 5])
def test_param_tensor(n_qubits: int, batch_size: int) -> None:
    for op in OPS_PARAM:
        if op in OPS_PARAM_1Q:
            supp: tuple = (random.randint(0, n_qubits - 1),)
        elif op in OPS_PARAM_2Q:
            supp = (random.randint(1, n_qubits - 1), 0)
        params = [f"th{i}" for i in range(op.n_params)]  # type: ignore [union-attr]
        values = {param: torch.rand(batch_size) for param in params}
        op_concrete = op(*supp, *params)
        psi_init = random_state(n_qubits)
        psi_star = op_concrete(psi_init, values)
        psi_expected = _calc_mat_vec_wavefunction(op_concrete, psi_init, values=values)
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
#     psi_expected = _calc_mat_vec_wavefunction(hamevo, psi)
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


# @pytest.mark.parametrize(
#     "projector, exp_projector_mat",
#     [
#         (
#             pyq.Projector(0, bra="1", ket="1"),
#             torch.tensor(
#                 [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
#                 dtype=torch.complex128,
#             ),
#         ),
#         (
#             pyq.N(0),
#             (IMAT - ZMAT) / 2.0,
#         ),
#         (
#             pyq.CNOT(0, 1),
#             torch.tensor(
#                 [
#                     [
#                         [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
#                         [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
#                         [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
#                         [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
#                     ]
#                 ],
#                 dtype=torch.complex128,
#             ),
#         ),
#     ],
# )
# def test_projector_tensor(
#     projector: Primitive, exp_projector_mat: torch.Tensor
# ) -> None:

#     nbqubits = int(log2(exp_projector_mat.shape[-1]))
#     projector_mat = projector.tensor(
#         n_qubits=nbqubits, values={"theta": torch.Tensor([1.0])}
#     ).squeeze(-1)
#     assert torch.allclose(projector_mat, exp_projector_mat, atol=1.0e-4)


# def test_circuit_tensor() -> None:
#     ops = [pyq.RX(0, "theta_0"), pyq.RY(0, "theta_1"), pyq.RX(1, "theta_2")]
#     circ = pyq.QuantumCircuit(2, ops)
#     values = {f"theta_{i}": torch.Tensor([float(i)]) for i in range(3)}
#     tensorcirc = circ.tensor(values)
#     assert tensorcirc.size() == (4, 4, 1)
#     assert torch.allclose(
#         tensorcirc,
#         torch.tensor(
#             [
#                 [
#                     [0.4742 + 0.0000j],
#                     [0.0000 - 0.7385j],
#                     [-0.2590 + 0.0000j],
#                     [0.0000 + 0.4034j],
#                 ],
#                 [
#                     [0.0000 - 0.7385j],
#                     [0.4742 + 0.0000j],
#                     [0.0000 + 0.4034j],
#                     [-0.2590 + 0.0000j],
#                 ],
#                 [
#                     [0.2590 + 0.0000j],
#                     [0.0000 - 0.4034j],
#                     [0.4742 + 0.0000j],
#                     [0.0000 - 0.7385j],
#                 ],
#                 [
#                     [0.0000 - 0.4034j],
#                     [0.2590 + 0.0000j],
#                     [0.0000 - 0.7385j],
#                     [0.4742 + 0.0000j],
#                 ],
#             ],
#             dtype=torch.complex128,
#         ),
#         atol=1.0e-4,
#     )
