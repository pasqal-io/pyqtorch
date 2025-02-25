from __future__ import annotations

import random

import pytest
import torch
from helpers import calc_mat_vec_wavefunction, get_op_support, random_pauli_hamiltonian

from pyqtorch.composite import Add, Scale, Sequence
from pyqtorch.hamiltonians import GeneratorType, HamiltonianEvolution
from pyqtorch.primitives import (
    CNOT,
    CPHASE,
    CRX,
    CRY,
    CRZ,
    CSWAP,
    CY,
    CZ,
    OPS_2Q,
    OPS_3Q,
    OPS_DIAGONAL,
    OPS_DIAGONAL_PARAM,
    OPS_DIGITAL,
    OPS_PARAM,
    OPS_PARAM_2Q,
    SWAP,
    N,
    Parametric,
    Primitive,
    Projector,
    Toffoli,
)
from pyqtorch.utils import (
    ATOL,
    RTOL,
    density_mat,
    permute_basis,
    random_state,
    todense_tensor,
)

pi = torch.tensor(torch.pi)


@pytest.mark.parametrize("n_qubits", [4, 5])
@pytest.mark.parametrize("make_param", [True, False])
@pytest.mark.parametrize("use_full_support", [True, False])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("diagonal", [False, True])
def test_primitive_tensor(
    n_qubits: int,
    make_param: bool,
    use_full_support: bool,
    batch_size: int,
    diagonal: bool,
) -> None:
    k_1q = 2 * n_qubits  # Number of 1-qubit terms
    k_2q = n_qubits**2  # Number of 2-qubit terms
    generator, param_list = random_pauli_hamiltonian(
        n_qubits, k_1q, k_2q, make_param, diagonal=diagonal
    )
    values = {param: torch.rand(1) for param in param_list}
    full_support = tuple(range(n_qubits)) if use_full_support else None
    generator_matrix = generator.tensor(
        values=values, full_support=full_support, diagonal=diagonal
    )
    generator_matrix = (
        generator_matrix.repeat((1, 1, batch_size))
        if len(generator_matrix.size()) == 3
        else generator_matrix.repeat((1, batch_size))
    )
    diag_op = diagonal and generator.is_diagonal
    primitive_op = Primitive(
        generator_matrix,
        qubit_support=full_support if use_full_support else generator.qubit_support,  # type: ignore[arg-type]
        diagonal=diagonal,
    )
    if diag_op:
        assert len(generator_matrix.size()) == 2
        assert primitive_op.is_diagonal
    else:
        assert len(generator_matrix.size()) == 3
        assert not primitive_op.is_diagonal
    assert torch.allclose(primitive_op.tensor(), generator_matrix, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("use_dm", [True, False])
@pytest.mark.parametrize("use_permute", [True, False])
@pytest.mark.parametrize("use_full_support", [True, False])
@pytest.mark.parametrize("n_qubits", [4, 5])
@pytest.mark.parametrize("batch_size", [1, 5])
def test_digital_tensor(
    n_qubits: int,
    batch_size: int,
    use_full_support: bool,
    use_permute: bool,
    use_dm: bool,
) -> None:
    """
    Goes through all non-parametric gates and tests their application to a random state
    in comparison with the `tensor` method, either using just the qubit support of the gate
    or expanding its matrix to the maximum qubit support of the full circuit.
    """
    op: type[Primitive]
    for op in OPS_DIGITAL:
        supp = get_op_support(op, n_qubits)
        op_concrete = op(*supp)
        if op in OPS_DIAGONAL:
            assert op_concrete.is_diagonal
        psi_init = random_state(n_qubits, batch_size)
        if use_dm:
            psi_star = op_concrete(density_mat(psi_init))
        else:
            psi_star = op_concrete(psi_init)
        full_support = tuple(range(n_qubits)) if use_full_support else None
        psi_expected = calc_mat_vec_wavefunction(
            op_concrete, psi_init, full_support=full_support, use_permute=use_permute
        )
        if use_dm:
            psi_expected = density_mat(psi_expected)
        assert torch.allclose(psi_star, psi_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("use_dm", [True, False])
@pytest.mark.parametrize("use_permute", [True, False])
@pytest.mark.parametrize("use_full_support", [True, False])
@pytest.mark.parametrize("n_qubits", [4, 5])
@pytest.mark.parametrize("batch_size", [1, 5])
def test_param_tensor(
    n_qubits: int,
    batch_size: int,
    use_full_support: bool,
    use_permute: bool,
    use_dm: bool,
) -> None:
    """
    Goes through all parametric gates and tests their application to a random state
    in comparison with the `tensor` method, either using just the qubit support of the gate
    or expanding its matrix to the maximum qubit support of the full circuit.
    """
    op: type[Parametric]
    for op in OPS_PARAM:
        supp = get_op_support(op, n_qubits)
        params = [f"th{i}" for i in range(op.n_params)]
        op_concrete = op(*supp, *params)
        if op in OPS_DIAGONAL_PARAM:
            assert op_concrete.is_diagonal
        psi_init = random_state(n_qubits)
        values = {param: torch.rand(batch_size) for param in params}
        if use_dm:
            psi_star = op_concrete(density_mat(psi_init), values)
        else:
            psi_star = op_concrete(psi_init, values)
        full_support = tuple(range(n_qubits)) if use_full_support else None
        psi_expected = calc_mat_vec_wavefunction(
            op_concrete,
            psi_init,
            values=values,
            full_support=full_support,
            use_permute=use_permute,
        )
        if use_dm:
            psi_expected = density_mat(psi_expected)
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
        supp = get_op_support(op, n_qubits)
        op_concrete = Scale(op(*supp), torch.rand(1))
        if op in OPS_DIAGONAL:
            assert op_concrete.is_diagonal
        op_list.append(op_concrete)
    for op in OPS_PARAM:
        supp = get_op_support(op, n_qubits)
        params = [f"{op.__name__}_th{i}" for i in range(op.n_params)]
        values.update({param: torch.rand(batch_size) for param in params})
        op_concrete = op(*supp, *params)
        if op in OPS_DIAGONAL_PARAM:
            assert op_concrete.is_diagonal
        op_list.append(op_concrete)
    random.shuffle(op_list)
    op_composite = compose(op_list)
    psi_init = random_state(n_qubits, batch_size)
    psi_star = op_composite(psi_init, values)
    full_support = tuple(range(n_qubits)) if use_full_support else None
    psi_expected = calc_mat_vec_wavefunction(
        op_composite, psi_init, values=values, full_support=full_support
    )
    assert torch.allclose(psi_star, psi_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("use_full_support", [True, False])
@pytest.mark.parametrize("n_qubits", [4, 5])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("compose", [Sequence, Add])
def test_diagonal_sequence_tensor(
    n_qubits: int,
    batch_size: int,
    use_full_support: bool,
    compose: type[Sequence] | type[Add],
) -> None:
    op_list = []
    values = {}
    op: type[Primitive] | type[Parametric]
    for op in OPS_DIAGONAL.intersection(OPS_DIGITAL):
        supp = get_op_support(op, n_qubits)
        op_concrete = Scale(op(*supp), torch.rand(1))
        op_list.append(op_concrete)
    for op in OPS_DIAGONAL_PARAM.intersection(OPS_PARAM):
        supp = get_op_support(op, n_qubits)
        params = [f"{op.__name__}_th{i}" for i in range(op.n_params)]
        values.update({param: torch.rand(batch_size) for param in params})
        op_concrete = op(*supp, *params)
        op_list.append(op_concrete)
    random.shuffle(op_list)
    op_composite = compose(op_list)
    full_support = tuple(range(n_qubits)) if use_full_support else None
    assert op_composite.is_diagonal
    tensor_composite = op_composite.tensor(values, full_support=full_support)
    tensor_composite_diagonal = op_composite.tensor(
        values, full_support=full_support, diagonal=True
    )
    dense_diagonal = todense_tensor(tensor_composite_diagonal)
    assert torch.allclose(tensor_composite, dense_diagonal, rtol=RTOL, atol=ATOL)


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
        psi_expected = calc_mat_vec_wavefunction(
            op, psi_init, full_support=full_support
        )
        assert torch.allclose(psi_star, psi_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n_qubits", [4, 5])
@pytest.mark.parametrize("operator", [N, CNOT, SWAP])
@pytest.mark.parametrize("use_full_support", [True, False])
def test_projector_vs_operator(
    n_qubits: int,
    operator: type[Primitive],
    use_full_support: bool,
) -> None:
    if operator == N:
        supp: tuple = (random.randint(0, n_qubits - 1),)
        op_concrete = N(*supp)
        projector = Projector(supp, "1", "1")
    if operator == CNOT:
        supp = tuple(random.sample(range(n_qubits), 2))
        op_concrete = CNOT(*supp)
        projector = Add(
            [
                Projector(supp, "00", "00"),
                Projector(supp, "01", "01"),
                Projector(supp, "10", "11"),
                Projector(supp, "11", "10"),
            ]
        )
    if operator == SWAP:
        supp = tuple(random.sample(range(n_qubits), 2))
        op_concrete = SWAP(*supp)
        projector = Add(
            [
                Projector(supp, "00", "00"),
                Projector(supp, "01", "10"),
                Projector(supp, "10", "01"),
                Projector(supp, "11", "11"),
            ]
        )
    full_support = tuple(range(n_qubits)) if use_full_support else None
    projector_mat = projector.tensor(full_support=full_support)
    operator_mat = op_concrete.tensor(full_support=full_support)
    assert torch.allclose(projector_mat, operator_mat, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n_qubits", [4, 5])
@pytest.mark.parametrize("make_param", [True, False])
@pytest.mark.parametrize("use_full_support", [True, False])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("diagonal", [False, True])
def test_hevo_pauli_tensor(
    n_qubits: int,
    make_param: bool,
    use_full_support: bool,
    batch_size: int,
    diagonal: bool,
) -> None:
    k_1q = 2 * n_qubits  # Number of 1-qubit terms
    k_2q = n_qubits**2  # Number of 2-qubit terms
    generator, param_list = random_pauli_hamiltonian(
        n_qubits, k_1q, k_2q, make_param, diagonal=diagonal
    )

    values = {param: torch.rand(batch_size) for param in param_list}
    psi_init = random_state(n_qubits, batch_size)
    full_support = tuple(range(n_qubits)) if use_full_support else None
    # Test the generator itself
    psi_star = generator(psi_init, values)
    psi_expected = calc_mat_vec_wavefunction(generator, psi_init, values, full_support)
    assert torch.allclose(psi_star, psi_expected, rtol=RTOL, atol=ATOL)

    # Test the hamiltonian evolution
    tparam = "t"
    operator = HamiltonianEvolution(generator, tparam)
    if diagonal:
        assert generator.is_diagonal
        if not make_param:
            assert operator.is_diagonal

    if make_param:
        assert operator.generator_type == GeneratorType.PARAMETRIC_OPERATION
    else:
        assert operator.generator_type == GeneratorType.OPERATION
    values[tparam] = torch.rand(batch_size)
    psi_star = operator(psi_init, values)
    psi_expected = calc_mat_vec_wavefunction(operator, psi_init, values, full_support)
    assert torch.allclose(psi_star, psi_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("gen_qubits", [3, 4])
@pytest.mark.parametrize("n_qubits", [4, 5])
@pytest.mark.parametrize("use_full_support", [True, False])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("diagonal", [False, True])
def test_hevo_tensor_tensor(
    gen_qubits: int,
    n_qubits: int,
    use_full_support: bool,
    batch_size: int,
    diagonal: bool,
) -> None:
    k_1q = 2 * gen_qubits  # Number of 1-qubit terms
    k_2q = gen_qubits**2  # Number of 2-qubit terms
    generator, _ = random_pauli_hamiltonian(gen_qubits, k_1q, k_2q, diagonal=diagonal)
    psi_init = random_state(n_qubits, batch_size)
    full_support = tuple(range(n_qubits)) if use_full_support else None
    # Test the hamiltonian evolution
    generator_matrix = generator.tensor()
    supp = tuple(random.sample(range(n_qubits), gen_qubits))
    tparam = "t"
    operator = HamiltonianEvolution(generator_matrix, tparam, supp)
    assert operator.generator_type == GeneratorType.TENSOR
    if diagonal:
        assert generator.is_diagonal
        assert operator.is_diagonal
    values = {tparam: torch.rand(batch_size)}
    psi_star = operator(psi_init, values)
    psi_expected = calc_mat_vec_wavefunction(operator, psi_init, values, full_support)
    assert torch.allclose(psi_star, psi_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n_qubits", [3, 5])
@pytest.mark.parametrize("diagonal", [False, True])
def test_permute_tensor(n_qubits: int, diagonal: bool) -> None:
    ops = OPS_2Q.union(OPS_3Q) if not diagonal else {CZ}
    for op in ops:
        supp, ordered_supp = get_op_support(op, n_qubits, get_ordered=True)

        op_concrete1 = op(*supp)
        op_concrete2 = op(*ordered_supp)

        mat1 = op_concrete1.tensor(diagonal=diagonal)
        mat2 = op_concrete2.tensor(diagonal=diagonal)

        perm = op_concrete1._qubit_support.qubits

        assert torch.allclose(
            mat1, permute_basis(mat2, perm, inv=True, diagonal=diagonal)
        )
        assert torch.allclose(mat2, permute_basis(mat1, perm, diagonal=diagonal))


@pytest.mark.parametrize("n_qubits", [3, 5])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("diagonal", [False, True])
def test_permute_tensor_parametric(
    n_qubits: int, batch_size: int, diagonal: bool
) -> None:
    ops = (
        OPS_PARAM_2Q if not diagonal else OPS_PARAM_2Q.intersection(OPS_DIAGONAL_PARAM)
    )
    for op in ops:
        supp, ordered_supp = get_op_support(op, n_qubits, get_ordered=True)
        params = [f"{op.__name__}_th{i}" for i in range(op.n_params)]
        values = {param: torch.rand(batch_size) for param in params}

        op_concrete1 = op(*supp, *params)
        op_concrete2 = op(*ordered_supp, *params)

        mat1 = op_concrete1.tensor(values=values, diagonal=diagonal)
        mat2 = op_concrete2.tensor(values=values, diagonal=diagonal)

        perm = op_concrete1._qubit_support.qubits

        assert torch.allclose(
            mat1, permute_basis(mat2, perm, inv=True, diagonal=diagonal)
        )
        assert torch.allclose(mat2, permute_basis(mat1, perm, diagonal=diagonal))


def test_tensor_symmetries() -> None:
    assert not torch.allclose(CNOT(0, 1).tensor(), CNOT(1, 0).tensor())
    assert not torch.allclose(CY(0, 1).tensor(), CY(1, 0).tensor())
    assert not torch.allclose(CZ(0, 1).tensor(), CY(1, 0).tensor())
    assert not torch.allclose(CRX(0, 1, 1.0).tensor(), CRX(1, 0, 1.0).tensor())
    assert not torch.allclose(CRY(0, 1, 1.0).tensor(), CRY(1, 0, 1.0).tensor())
    assert not torch.allclose(CRZ(0, 1, 1.0).tensor(), CRZ(1, 0, 1.0).tensor())
    assert torch.allclose(CPHASE(0, 1, 1.0).tensor(), CPHASE(1, 0, 1.0).tensor())
    assert torch.allclose(SWAP(0, 1).tensor(), SWAP(1, 0).tensor())
    assert torch.allclose(CSWAP(0, (1, 2)).tensor(), CSWAP(0, (2, 1)).tensor())
    assert torch.allclose(CSWAP(1, (0, 2)).tensor(), CSWAP(1, (2, 0)).tensor())
    assert torch.allclose(CSWAP(2, (0, 1)).tensor(), CSWAP(2, (1, 0)).tensor())
    assert torch.allclose(Toffoli((0, 1), 2).tensor(), Toffoli((1, 0), 2).tensor())
    assert torch.allclose(Toffoli((0, 2), 1).tensor(), Toffoli((2, 0), 1).tensor())
    assert torch.allclose(Toffoli((1, 2), 0).tensor(), Toffoli((2, 1), 0).tensor())
