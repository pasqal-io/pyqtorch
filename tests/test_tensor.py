from __future__ import annotations

import random

import pytest
import torch
from conftest import _calc_mat_vec_wavefunction

from pyqtorch.analog import Add, Scale
from pyqtorch.circuit import Sequence
from pyqtorch.parametric import OPS_PARAM, OPS_PARAM_1Q, OPS_PARAM_2Q, Parametric
from pyqtorch.primitive import (
    OPS_1Q,
    OPS_2Q,
    OPS_3Q,
    OPS_DIGITAL,
    Primitive,
    Projector,
    Toffoli,
)
from pyqtorch.utils import (
    ATOL,
    RTOL,
    random_state,
)

pi = torch.tensor(torch.pi)


def _get_op_support(op: type[Primitive] | type[Parametric], n_qubits: int) -> tuple:
    """Decides a random qubit support for any gate, up to a some max n_qubits."""
    if op in OPS_1Q.union(OPS_PARAM_1Q):
        supp: tuple = (random.randint(0, n_qubits - 1),)
    elif op in OPS_2Q.union(OPS_PARAM_2Q):
        supp = tuple(random.sample(range(n_qubits), 2))
    elif op in OPS_3Q:
        i, j, k = tuple(random.sample(range(n_qubits), 3))
        supp = ((i, j), k) if op == Toffoli else (i, (j, k))
    return supp


@pytest.mark.parametrize("use_full_support", [True, False])
@pytest.mark.parametrize("n_qubits", [4, 5])
@pytest.mark.parametrize("batch_size", [1, 5])
def test_digital_tensor(n_qubits: int, batch_size: int, use_full_support: bool) -> None:
    """
    Goes through all non-parametric gates and tests their application to a random state
    in comparison with the `tensor` method, either using just the qubit support of the gate
    or expanding its matrix to the maximum qubit support of the full circuit.
    """
    op: type[Primitive]
    for op in OPS_DIGITAL:
        supp = _get_op_support(op, n_qubits)
        op_concrete = op(*supp)
        psi_init = random_state(n_qubits, batch_size)
        psi_star = op_concrete(psi_init)
        full_support = tuple(range(n_qubits)) if use_full_support else None
        psi_expected = _calc_mat_vec_wavefunction(
            op_concrete, psi_init, full_support=full_support
        )
        assert torch.allclose(psi_star, psi_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("use_full_support", [True, False])
@pytest.mark.parametrize("n_qubits", [4, 5])
@pytest.mark.parametrize("batch_size", [1, 5])
def test_param_tensor(n_qubits: int, batch_size: int, use_full_support: bool) -> None:
    """
    Goes through all parametric gates and tests their application to a random state
    in comparison with the `tensor` method, either using just the qubit support of the gate
    or expanding its matrix to the maximum qubit support of the full circuit.
    """
    op: type[Parametric]
    for op in OPS_PARAM:
        supp = _get_op_support(op, n_qubits)
        params = [f"th{i}" for i in range(op.n_params)]
        op_concrete = op(*supp, *params)
        psi_init = random_state(n_qubits)
        values = {param: torch.rand(batch_size) for param in params}
        psi_star = op_concrete(psi_init, values)
        full_support = tuple(range(n_qubits)) if use_full_support else None
        psi_expected = _calc_mat_vec_wavefunction(
            op_concrete, psi_init, values=values, full_support=full_support
        )
        assert torch.allclose(psi_star, psi_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("use_full_support", [True, False])
@pytest.mark.parametrize("n_qubits", [4, 5])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("compose", [Sequence, Add])
def test_sequence_tensor(
    n_qubits: int,
    batch_size: int,
    use_full_support: bool,
    compose: type[Sequence] | type[Add],
) -> None:
    op_list = []
    values = {}
    op: type[Primitive] | type[Parametric]
    """
    Builds a Sequence or Add composition of all possible gates on random qubit
    supports. Also assigns a Scale of a random value to the non-parametric gates.
    Tests the forward method (which goes through each gate individually) to the
    `tensor` method, which builds the full operator matrix and applies it.
    """
    for op in OPS_DIGITAL:
        supp = _get_op_support(op, n_qubits)
        op_concrete = Scale(op(*supp), torch.rand(1))
        op_list.append(op_concrete)
    for op in OPS_PARAM:
        supp = _get_op_support(op, n_qubits)
        params = [f"{op.__name__}_th{i}" for i in range(op.n_params)]
        values.update({param: torch.rand(batch_size) for param in params})
        op_concrete = op(*supp, *params)
        op_list.append(op_concrete)
    random.shuffle(op_list)
    op_composite = compose(op_list)
    psi_init = random_state(n_qubits, batch_size)
    psi_star = op_composite(psi_init, values)
    full_support = tuple(range(n_qubits)) if use_full_support else None
    psi_expected = _calc_mat_vec_wavefunction(
        op_composite, psi_init, values=values, full_support=full_support
    )
    assert torch.allclose(psi_star, psi_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("use_full_support", [True, False])
@pytest.mark.parametrize("n_qubits", [4, 5])
@pytest.mark.parametrize("n_proj", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 5])
def test_projector_tensor(
    n_qubits: int, n_proj: int, batch_size: int, use_full_support: bool
) -> None:
    """
    Instantiates various random projectors on arbitrary qubit support
    and compares the forward method with directly applying the tensor.
    """
    iterations = 5
    for _ in range(iterations):
        rand_int_1 = random.randint(0, 2**n_proj - 1)
        rand_int_2 = random.randint(0, 2**n_proj - 1)
        bitstring1 = "{0:b}".format(rand_int_1).zfill(n_proj)
        bitstring2 = "{0:b}".format(rand_int_2).zfill(n_proj)
        supp = tuple(random.sample(range(n_qubits), n_proj))
        op = Projector(supp, bitstring1, bitstring2)
        psi_init = random_state(n_qubits, batch_size)
        psi_star = op(psi_init)
        full_support = tuple(range(n_qubits)) if use_full_support else None
        psi_expected = _calc_mat_vec_wavefunction(
            op, psi_init, full_support=full_support
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
#     generator = generator.tensor()
#     generator = generator + torch.conj(torch.transpose(generator, 0, 1))
#     hamevo = pyq.HamiltonianEvolution(
#         generator, vparam, tuple(range(n_qubits)), parametric
#     )
#     assert hamevo.generator_type == GeneratorType.TENSOR
#     vals = {vparam: torch.tensor([0.5])}
#     psi = random_state(n_qubits)
#     psi_star = hamevo(psi, vals)
#     psi_expected = _calc_mat_vec_wavefunction(hamevo, psi, vals)
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
#     psi_expected = _calc_mat_vec_wavefunction(hamevo, psi, vals)
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
