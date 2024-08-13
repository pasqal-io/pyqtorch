from __future__ import annotations

import random

import torch

from pyqtorch.apply import apply_operator, apply_operator_permute
from pyqtorch.composite import Add, Scale, Sequence
from pyqtorch.primitives import (
    OPS_1Q,
    OPS_2Q,
    OPS_3Q,
    OPS_PARAM_1Q,
    OPS_PARAM_2Q,
    OPS_PAULI,
    Parametric,
    Primitive,
    Toffoli,
)


def calc_mat_vec_wavefunction(
    block: Primitive | Sequence,
    init_state: torch.Tensor,
    values: dict = dict(),
    full_support: tuple | None = None,
    use_permute: bool = False,
) -> torch.Tensor:
    """Get the result of applying the matrix representation of a block to an initial state.

    Args:
        block: The black operator to apply.
        init_state: Initial state to apply block on.
        values: Values of parameter if block is parametric.

    Returns:
        Tensor: The new wavefunction after applying the block.

    """
    mat = block.tensor(values=values, full_support=full_support)
    qubit_support = block.qubit_support if full_support is None else full_support
    apply_func = apply_operator_permute if use_permute else apply_operator
    return apply_func(
        init_state,
        mat,
        qubit_support,
    )


def get_op_support(op: type[Primitive] | type[Parametric], n_qubits: int) -> tuple:
    """Decides a random qubit support for any gate, up to a some max n_qubits."""
    if op in OPS_1Q.union(OPS_PARAM_1Q):
        supp: tuple = (random.randint(0, n_qubits - 1),)
    elif op in OPS_2Q.union(OPS_PARAM_2Q):
        supp = tuple(random.sample(range(n_qubits), 2))
    elif op in OPS_3Q:
        i, j, k = tuple(random.sample(range(n_qubits), 3))
        supp = ((i, j), k) if op == Toffoli else (i, (j, k))
    return supp


def random_pauli_hamiltonian(
    n_qubits: int,
    k_1q: int = 5,
    k_2q: int = 10,
    make_param: bool = False,
    default_scale_coeffs: float | None = None,
) -> tuple[Sequence, list]:
    """Creates a random Pauli Hamiltonian as a sum of k_1q + k_2q terms.

    Args:
        n_qubits (int): Number of qubits.
        k_1q (int, optional): Number of one-qubit terms. Defaults to 5.
        k_2q (int, optional): Number of two-qubit terms. Defaults to 10.
        make_param (bool, optional): Coefficients as parameters. Defaults to False.
        default_scale_coeffs (float | None, optional): Default value for the parameter
            of Scale operations. Defaults to None.

    Returns:
        tuple[Sequence, list]: Hamiltonian and list of parameters.
    """
    OPS_PAULI_choice = list(OPS_PAULI)
    one_q_terms: list = random.choices(OPS_PAULI_choice, k=k_1q)
    two_q_terms: list = [random.choices(OPS_PAULI_choice, k=2) for _ in range(k_2q)]
    terms: list = []
    for term in one_q_terms:
        supp = random.sample(range(n_qubits), 1)
        terms.append(term(supp))
    for term in two_q_terms:
        supp = random.sample(range(n_qubits), 2)
        terms.append(Sequence([term[0](supp[0]), term[1](supp[1])]))
    param_list = []
    for i, t in enumerate(terms):
        if random.random() > 0.5:
            if make_param:
                terms[i] = Scale(t, f"p_{i}")
                param_list.append(f"p_{i}")
            else:
                terms[i] = Scale(
                    t,
                    (
                        torch.rand(1)
                        if not default_scale_coeffs
                        else torch.tensor([default_scale_coeffs])
                    ),
                )
    return Add(terms), param_list
