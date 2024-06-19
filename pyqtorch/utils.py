from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache, partial, wraps
from logging import getLogger
from math import sqrt
from typing import Any, Callable, Sequence

import torch
from torch import Tensor

from pyqtorch.matrices import DEFAULT_MATRIX_DTYPE, DEFAULT_REAL_DTYPE

State = Tensor
Operator = Tensor

ATOL = 1e-06
RTOL = 0.0
GRADCHECK_ATOL = 1e-06

logger = getLogger(__name__)


def inner_prod(bra: Tensor, ket: Tensor) -> Tensor:
    """
    Compute the inner product :math:`\\langle\\bra|\\ket\\rangle`

    Arguments:
        bra: left part quantum state.
        ket: right part quantum state.

    Returns:
        The inner product.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Inner prod calculation")
        torch.cuda.nvtx.range_push("inner_prod")

    n_qubits = len(bra.size()) - 1
    bra = bra.reshape((2**n_qubits, bra.size(-1)))
    ket = ket.reshape((2**n_qubits, ket.size(-1)))
    res = torch.einsum("ib,ib->b", bra.conj(), ket)
    if logger.isEnabledFor(logging.DEBUG):
        torch.cuda.nvtx.range_pop()
        logger.debug("Inner prod complete")
    return res


def overlap(bra: Tensor, ket: Tensor) -> Tensor:
    """
    Compute the overlap :math:`|\\langle\\bra|\\ket\\rangle|^2`

    Arguments:
        bra: left part quantum state.
        ket: right part quantum state.

    Returns:
        The overlap.
    """
    return torch.pow(inner_prod(bra, ket).real, 2)


class StrEnum(str, Enum):
    def __str__(self) -> str:
        """Used when dumping enum fields in a schema."""
        ret: str = self.value
        return ret

    @classmethod
    def list(cls) -> list[str]:
        return list(map(lambda c: c.value, cls))  # type: ignore


class DiffMode(StrEnum):
    """
    Which Differentiation method to use.

    Options: Automatic Differentiation -  Using torch.autograd.
             Adjoint Differentiation   - An implementation of "Efficient calculation of gradients
                                       in classical simulations of variational quantum algorithms",
                                       Jones & Gacon, 2020
    """

    AD = "ad"
    ADJOINT = "adjoint"


def is_normalized(state: Tensor, atol: float = ATOL) -> bool:
    """
    Function to test if the probabilities from the state sum up to 1.

    Arguments:
        state: State to test.
        atol: Tolerance for check.

    Returns:
        True if normalized, False otherwise.
    """
    n_qubits = len(state.size()) - 1
    batch_size = state.size()[-1]
    state = state.reshape((2**n_qubits, batch_size))
    sum_probs = (state.abs() ** 2).sum(dim=0)
    ones = torch.ones(batch_size, dtype=DEFAULT_REAL_DTYPE)
    return torch.allclose(sum_probs, ones, rtol=RTOL, atol=atol)  # type: ignore[no-any-return]


def is_diag(H: Tensor, atol: Tensor = ATOL) -> bool:
    """
    Returns True if tensor H is diagonal.

    Reference: https://stackoverflow.com/questions/43884189/check-if-a-large-matrix-is-diagonal-matrix-in-python

    Arguments:
        H: Input tensor.
        atol: If off-diagonal values are lower than atol in amplitude, H is considered diagonal.

    Returns:
        True if diagonal, else False.
    """
    m = H.shape[0]
    p, q = H.stride()
    offdiag_view = torch.as_strided(H[:, 1:], (m - 1, m), (p + q, q))
    return torch.count_nonzero(torch.abs(offdiag_view).gt(atol)) == 0


def product_state(
    bitstring: str,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = DEFAULT_MATRIX_DTYPE,
) -> Tensor:
    """
    Create a batch of quantum states :math:`|\\b\\rangle` where b is the bitstring input.

    Arguments:
        bitstring: The bitstring b to represent.
        batch_size: The size of the batch.
        device: Device where tensors are stored.
        dtype: Type of tensors.

    Returns:
        Batch of quantum states representing bitstring.
    """
    state = torch.zeros((2 ** len(bitstring), batch_size), dtype=dtype)
    state[int(bitstring, 2)] = torch.tensor(1.0 + 0j, dtype=dtype)
    return state.reshape([2] * len(bitstring) + [batch_size]).to(device=device)


def zero_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = DEFAULT_MATRIX_DTYPE,
) -> Tensor:
    """
    Create a batch of :math:`|\\0\\rangle^{\otimes n}` states defined on n qubits.

    Arguments:
        n_qubits: Number of qubits n the state is defined on.
        batch_size: The size of the batch.
        device: Device where tensors are stored.
        dtype: Type of tensors.

    Returns:
        Batch of uniform quantum states.
    """
    return product_state("0" * n_qubits, batch_size, dtype=dtype, device=device)


def uniform_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = DEFAULT_MATRIX_DTYPE,
) -> Tensor:
    """
    Create a batch of uniform states with equal probabilities for computational basis.

    Arguments:
        n_qubits: Number of qubits the state is defined on.
        batch_size: The size of the batch.
        device: Device where tensors are stored.
        dtype: Type of tensors.

    Returns:
        Batch of uniform quantum states.
    """
    state = torch.ones((2**n_qubits, batch_size), dtype=dtype, device=device)
    state = state / torch.sqrt(torch.tensor(2**n_qubits))
    return state.reshape([2] * n_qubits + [batch_size])


def random_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = DEFAULT_MATRIX_DTYPE,
    generator: torch.Generator = None,
) -> Tensor:
    """
    Create a batch of random quantum state.

    Arguments:
        n_qubits: Number of qubits the state is defined on.
        batch_size: The size of the batch.
        device: Device where tensors are stored.
        dtype: Type of tensors.

    Returns:
        Batch of random quantum states.
    """

    def _normalize(wf: Tensor) -> Tensor:
        return wf / torch.sqrt((wf.abs() ** 2).sum())

    def _rand(n_qubits: int) -> Tensor:
        N = 2**n_qubits
        x = -torch.log(torch.rand(N, generator=generator, dtype=dtype))
        sumx = torch.sum(x)
        phases = torch.rand(N, generator=generator, dtype=dtype) * 2.0 * torch.pi
        return _normalize(
            (torch.sqrt(x / sumx) * torch.exp(1j * phases))
            .reshape(N, 1)
            .type(dtype)
            .to(device)
        )

    state = torch.concat(tuple(_rand(n_qubits) for _ in range(batch_size)), dim=1)
    return state.reshape([2] * n_qubits + [batch_size]).to(device=device)


def param_dict(keys: Sequence[str], values: Sequence[Tensor]) -> dict[str, Tensor]:
    """
    Create a dictionary mapping parameters with their values.

    Arguments:
        keys: Parameter names as keys.
        values: Values of parameters.

    Returns:
        Dictionary mapping parameters and values.
    """
    return {key: val for key, val in zip(keys, values)}


class DensityMatrix(Tensor):
    pass


def density_mat(state: Tensor) -> DensityMatrix:
    """
    Computes the density matrix from a pure state vector.

    Arguments:
        state: The pure state vector :math:`|\\psi\\rangle`.

    Returns:
        Tensor: The density matrix :math:`\\rho = |\psi \\rangle \\langle\\psi|`.
    """
    n_qubits = len(state.size()) - 1
    batch_size = state.shape[-1]
    state = state.reshape(2**n_qubits, batch_size)
    return DensityMatrix(torch.einsum("ib,jb->ijb", (state, state.conj())))


def operator_kron(op1: Tensor, op2: Tensor) -> Tensor:
    """
    Compute the Kronecker product of two operators.

    Prevents errors related to the shape of the operators
    [2**n_qubits, 2**n_qubits, batch_size] when simply using `torch.kron()`.
    Use of `.contiguous()` to avoid errors related to the `torch.kron()` of a transposing tensor

    Arguments:
        op1: The first input tensor.
        op2: The second input tensor.

    Returns:
        The resulting tensor after applying the Kronecker product
    """
    batch_size_1, batch_size_2 = op1.size(2), op2.size(2)
    if batch_size_1 > batch_size_2:
        op2 = op2.repeat(1, 1, batch_size_1)[:, :, :batch_size_1]
    elif batch_size_2 > batch_size_1:
        op1 = op1.repeat(1, 1, batch_size_2)[:, :, :batch_size_2]
    kron_product = torch.einsum("ikb,jlb->ijklb", op1.contiguous(), op2.contiguous())
    return kron_product.reshape(
        op1.size(0) * op2.size(0), op1.size(1) * op2.size(1), op1.size(2)
    )


def promote_operator(operator: Tensor, target: int, n_qubits: int) -> Tensor:
    from pyqtorch.primitive import I

    """
    Promotes `operator` to the size of the circuit (number of qubits and batch).
    Targeting the first qubit implies target = 0, so target > n_qubits - 1.

    Arguments:
        operator: The operator tensor to be promoted.
        target: The index of the target qubit to which the operator is applied.
            Targeting the first qubit implies target = 0, so target > n_qubits - 1.
        n_qubits: Number of qubits in the circuit.

    Returns:
        Tensor: The promoted operator tensor.

    Raises:
        ValueError: If `target` is outside the valid range of qubits.
    """
    if target > n_qubits - 1:
        raise ValueError(
            "The target must be a valid qubit index within the circuit's range."
        )
    qubits = torch.arange(0, n_qubits)
    qubits = qubits[qubits != target]
    for qubit in qubits:
        operator = torch.where(
            target > qubit,
            operator_kron(I(target).unitary(), operator),
            operator_kron(operator, I(target).unitary()),
        )
    return operator


def random_dm_promotion(
    target: int, dm_input: DensityMatrix, n_qubits: int
) -> DensityMatrix:
    """
    Promotes the density matrix `dm_input` at the specified target qubit with random states.

    Args:
        target (int): The index of the target qubit where the promotion will be applied.
        dm_input (DensityMatrix): The input density matrix to promote.
        n_qubits (int): The total number of qubits in the quantum system.

    Returns:
        DensityMatrix: The density matrix after applying random promotion.

    Raises:
        ValueError: If `target` is not within the valid range [0, n_qubits - 1].
    """
    if target > n_qubits - 1:
        raise ValueError(
            "The target must be a valid qubit index within the circuit range."
        )
    rng = torch.Generator().manual_seed(n_qubits)
    if target == 0 or target == n_qubits - 1:
        state_random: Tensor = random_state(n_qubits - 1, generator=rng)
        dm_random: DensityMatrix = density_mat(state_random)
        dm_input = torch.where(
            torch.tensor(target) == 0,
            operator_kron(dm_input, dm_random),
            operator_kron(dm_random, dm_input),
        )
    else:
        state_random_1 = random_state(target, generator=rng)
        state_random_2 = random_state(n_qubits - (target + 1), generator=rng)
        dm_random_1, dm_random_2 = density_mat(state_random_1), density_mat(
            state_random_2
        )
        dm_input = operator_kron(dm_random_1, operator_kron(dm_input, dm_random_2))
    return dm_input


def add_batch_dim(operator: Tensor, batch_size: int = 1) -> Tensor:
    """In case we have a sequence of batched parametric gates mixed with primitive gates,
    we adjust the batch_dim of the primitive gates to match."""
    return (
        operator.repeat(1, 1, batch_size)
        if operator.shape != (2, 2, batch_size)
        else operator
    )


def operator_to_sparse_diagonal(operator: Tensor) -> Tensor:
    """Convert operator to a sparse diagonal tensor.
    Arguments:
        operator: Operator to convert.
    Returns:
        A sparse tensor.
    """
    operator = torch.diag(operator)
    indices, values, size = (
        torch.nonzero(operator),
        operator[operator != 0],
        len(operator),
    )
    indices = torch.stack((indices.flatten(), indices.flatten()))
    return torch.sparse_coo_tensor(indices, values, (size, size))


def cache(func: Callable, *, maxsize: int = 1) -> Callable:
    """Cache a function returning a tensor by memoizing its most recent calls.

    This decorator extends `methodtools.lru_cache` to also cache a function on
    PyTorch grad mode status (enabled or disabled). This prevents cached tensors
    detached from the graph (for example computed within a `with torch.no_grad()`
    block) from being used by mistake by later code which requires tensors attached
    to the graph.

    By default, the cache size is `1`, which means that only the most recent call is
    cached. Use the `maxsize` keyword argument to change the maximum cache size.

    Warning:
        This decorator should only be used for PyTorch tensors.

    Args:
        func: Function returning a tensor, can take any number of arguments.

    Returns:
        Cached function.
    """
    if func is None:
        return partial(cache, maxsize=maxsize)

    # define a function cached on its arguments and also on PyTorch grad mode status
    @lru_cache(maxsize=maxsize)
    def grad_cached_func(*args: Any, grad_enabled: bool, **kwargs: Any) -> Tensor:
        return func(*args, **kwargs)

    # wrap `func` to call its modified cached version
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Tensor:
        return grad_cached_func(*args, grad_enabled=torch.is_grad_enabled(), **kwargs)

    return wrapper


def hairer_norm(x: Tensor) -> Tensor:
    """Rescaled Frobenius norm of a batched matrix.

    Args:
        x: Tensor of shape `(..., n, n)`.

    Returns:
        Tensor of shape `(...)` holding the norm of each matrix in the batch.
    """
    return torch.linalg.matrix_norm(x) / sqrt(x.size(-1) * x.size(-2))


@dataclass
class Result:
    states: Tensor


class SolverType(StrEnum):
    DP5_SE = "dp5_se"
    """Uses fifth-order Dormand-Prince Schrodinger equation solver"""

    DP5_ME = "dp5_me"
    """Uses fifth-order Dormand-Prince master equation solver"""

    KRYLOV_SE = "krylov_se"
    """Uses Krylov Schrodinger equation solver"""
