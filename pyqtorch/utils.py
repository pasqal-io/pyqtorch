from __future__ import annotations

import logging
from collections import Counter, OrderedDict
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache, partial, wraps
from logging import getLogger
from math import sqrt
from string import ascii_uppercase as ABC
from typing import Any, Callable, Sequence

import numpy as np
import torch
from numpy import arange, argsort, array, delete, log2
from numpy import ndarray as NDArray
from torch import Tensor, moveaxis
from typing_extensions import TypeAlias

import pyqtorch as pyq
from pyqtorch.matrices import DEFAULT_MATRIX_DTYPE, DEFAULT_REAL_DTYPE, IDIAG, IMAT

State: TypeAlias = Tensor
Operator: TypeAlias = Tensor

ATOL = 1e-06
ATOL_embedding = 1e-03
RTOL = 0.0
GRADCHECK_ATOL = 1e-04
GRADCHECK_ATOL_hamevo = 1e-03
GRADCHECK_sampling_ATOL = 1e-01
PSR_ACCEPTANCE = 1e-05
ABC_ARRAY: NDArray = array(list(ABC))

logger = getLogger(__name__)


class OrderedCounter(Counter, OrderedDict):  # type: ignore [misc]
    pass


def qubit_support_as_tuple(support: int | tuple[int, ...]) -> tuple[int, ...]:
    """Make sure support returned is a tuple of integers.

    Args:
        support (int | tuple[int, ...]): Qubit support.

    Returns:
        tuple[int, ...]: Qubit support as tuple.
    """
    if isinstance(support, np.integer):
        return (support.item(),)
    qubit_support = (support,) if isinstance(support, int) else tuple(support)
    return qubit_support


def _round_operator(t: Tensor, decimals: int = 4) -> Tensor:
    if torch.is_complex(t):

        def _round(_t: Tensor) -> Tensor:
            r = _t.real.round(decimals=decimals)
            i = _t.imag.round(decimals=decimals)
            return torch.complex(r, i)

    else:

        def _round(_t: Tensor) -> Tensor:
            return _t.round(decimals=decimals)

    fn = torch.vmap(_round)
    return fn(t)


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


def counts_to_orderedcounter(
    binary_count: Tensor, length_bitstring: int
) -> OrderedCounter:
    """Convert counts (from torch.bincount) to an OrderedCounter.

    Note the output of torch.bincount is ordered
    and having an OrderedCounter can be convenient.

    Args:
        binary_count (Tensor): Counts per bitstring as Tensor.
        length_bitstring (int): Length of the bitstring.

    Returns:
        OrderedCounter: Ordered counter of bitstrings
    """
    return OrderedCounter(
        OrderedDict(
            [
                (format(k, "0{}b".format(length_bitstring)), count.item())
                for k, count in enumerate(binary_count)
                if count > 0
            ]
        )
    )


def sample_multinomial(
    probs: Tensor,
    length_bitstring: int,
    n_samples: int,
    return_counter: bool = True,
    minlength: int = 0,
) -> OrderedCounter | Tensor:
    """Sample bitstrings from a probability distribution.

    Args:
        probs (Tensor): Probability distribution
        length_bitstring (int): Maximal length of bitstring.
        n_samples (int): Number of samples to extract.
        instead of ratios.
        return_counter (bool): If True, return OrderedCounter object.
            Otherwise, the result of torch.bincount is returned.
        minlength (int): minimum number of bins. Should be non-negative.

    Returns:
        OrderedCounter | Tensor: Sampled bitstrings with their frequencies or probabilities.
    """

    bincount_output = torch.bincount(
        torch.multinomial(input=probs, num_samples=n_samples, replacement=True),
        minlength=minlength,
    )

    if return_counter:
        return counts_to_orderedcounter(bincount_output, length_bitstring)
    return bincount_output


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
    """

    AD = "ad"
    """Use torch.autograd to perform differentiation."""
    ADJOINT = "adjoint"
    """An implementation of "Efficient calculation of gradients
                                       in classical simulations of variational quantum algorithms",
                                       Jones & Gacon, 2020"""

    GPSR = "gpsr"
    """The generalized parameter-shift rule"""


class DropoutMode(StrEnum):
    """
    Which Dropout mode to use, using the methods stated in https://arxiv.org/abs/2310.04120.

    Options: rotational    - Randomly drops entangling rotational gates.
             entangling    - Randomly drops entangling gates.
             canonical_fwd - Randomly drops rotational gates and next immediate entangling
                            gates whose target bit is located on dropped rotational gates.
    """

    ROTATIONAL = "rotational_dropout"
    ENTANGLING = "entangling_dropout"
    CANONICAL_FWD = "canonical_fwd_dropout"


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


def is_diag_batched(H: Operator, atol: Tensor = ATOL, batch_dim: int = -1) -> bool:
    """
    Returns True if the batched tensors of H are diagonal.
    Arguments:
        H: Input tensors.
        atol: Tolerance for near-zero values.
        batch_dim: batch dimension to go over.
    Returns:
        True if diagonal, else False.
    """
    if len(H.shape) > 2:
        diag_check = torch.tensor(
            [is_diag(H[..., i], atol) for i in range(H.shape[batch_dim])],
            device=H.device,
        )
        return bool(torch.prod(diag_check).item())
    else:
        return is_diag(H, atol)


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
    offdiag_view = H - torch.diag(torch.diag(H))
    return torch.count_nonzero(torch.abs(offdiag_view).gt(atol)).item() == 0


def finitediff(
    f: Callable,
    x: Tensor,
    derivative_indices: tuple[int, ...],
    eps: float | None = None,
) -> Tensor:
    """
    Compute the finite difference of a function at a point.

    Args:
        f: The function to differentiate.
        x: Input of size `(batch_size, input_size)`.
        derivative_indices: Which *input* to differentiate (i.e. which variable x[:,i])
        eps: finite difference spacing (uses `torch.finfo(x.dtype).eps ** (1 / (2 + order))`
            as default)

    Returns:
        (Tensor): The finite difference of the function at the point `x`.
    """

    if eps is None:
        order = len(derivative_indices)
        eps = torch.finfo(x.dtype).eps ** (1 / (2 + order))

    # compute derivative direction vector(s)
    delta = torch.zeros_like(x)
    i = derivative_indices[0]
    delta[:, i] += torch.as_tensor(eps, dtype=x.dtype)
    denominator = 1 / delta[:, i]

    # recursive finite differencing for higher order than 3 / mixed derivatives
    if len(derivative_indices) > 3 or len(set(derivative_indices)) > 1:
        di = derivative_indices[1:]
        return (
            (finitediff(f, x + delta, di) - finitediff(f, x - delta, di))
            * denominator
            / 2
        )
    if len(derivative_indices) == 3:
        return (
            (f(x + 2 * delta) - 2 * f(x + delta) + 2 * f(x - delta) - f(x - 2 * delta))
            * denominator**3
            / 2
        )
    if len(derivative_indices) == 2:
        return (f(x + delta) + f(x - delta) - 2 * f(x)) * denominator**2
    if len(derivative_indices) == 1:
        return (f(x + delta) - f(x - delta)) * denominator / 2
    raise ValueError(
        "If you see this error there is a bug in the `finitediff` function."
    )


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
    Create a batch of :math:`|\\0\\rangle^{\\otimes n}` states defined on n qubits.

    Args:
        n_qubits (int): Number of qubits n the state is defined on.
        batch_size (int, optional): The size of the batch.
            Defaults to 1.
        device (str | torch.device, optional): Device where tensors are stored.
            Defaults to "cpu".
        dtype (torch.dtype, optional): Type of tensors. Defaults to DEFAULT_MATRIX_DTYPE.

    Returns:
        Tensor: Batch of uniform quantum states.
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
        Tensor: The density matrix :math:`\\rho = |\\psi \\rangle \\langle\\psi|`.
    """
    if isinstance(state, DensityMatrix):
        return state
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


def expand_operator(
    operator: Tensor,
    qubit_support: tuple[int, ...],
    full_support: tuple[int, ...],
    diagonal: bool = False,
) -> Tensor:
    """
    Expands an operator acting on a given qubit_support to act on a larger full_support
    by explicitly filling in identity matrices on all remaining qubits.
    """
    full_support = tuple(sorted(full_support))
    qubit_support = tuple(sorted(qubit_support))
    if not set(qubit_support).issubset(set(full_support)):
        raise ValueError(
            "Expanding tensor operation requires a `full_support` argument "
            "larger than or equal to the `qubit_support`."
        )
    device, dtype = operator.device, operator.dtype
    if not diagonal:
        other = IMAT.clone().to(device=device, dtype=dtype).unsqueeze(2)
    else:
        other = IDIAG.clone().to(device=device, dtype=dtype).unsqueeze(1)

    for i in set(full_support) - set(qubit_support):
        qubit_support += (i,)
        operator = torch.kron(operator.contiguous(), other)
    operator = permute_basis(operator, qubit_support, inv=True, diagonal=diagonal)
    return operator


def promote_operator(operator: Tensor, target: int, n_qubits: int) -> Tensor:
    from pyqtorch.primitives import I

    """
    FIXME: Remove and replace usage with the `expand_operator` above.

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
            operator_kron(I(target).tensor(), operator),
            operator_kron(operator, I(target).tensor()),
        )
    return operator


def permute_state(
    state: Tensor, qubit_support: tuple | list, inv: bool = False
) -> Tensor:
    """Takes a state tensor and permutes the qubit amplitudes
    according to the order of the qubit support.

    Args:
        state (Tensor): State to permute over.
        qubit_support (tuple): Qubit support.
        inv (bool): Applies the inverse permutation instead.

    Returns:
        Tensor: Permuted state.
    """
    if tuple(qubit_support) == tuple(sorted(qubit_support)):
        return state

    ordered_support = argsort(qubit_support)
    ranked_support = argsort(ordered_support)

    perm = list(ranked_support) + [len(qubit_support)]

    if inv:
        perm = np.argsort(perm).tolist()

    return state.permute(perm)


def permute_basis(
    operator: Tensor,
    qubit_support: tuple,
    inv: bool = False,
    diagonal: bool = False,
) -> Tensor:
    """Takes an operator tensor and permutes the rows and
    columns according to the order of the qubit support.

    Args:
        operator (Tensor): Operator to permute over.
        qubit_support (tuple): Qubit support.
        inv (bool): Applies the inverse permutation instead.

    Returns:
        Tensor: Permuted operator.
    """
    ordered_support = argsort(qubit_support)
    ranked_support = argsort(ordered_support)
    n_qubits = len(qubit_support)
    if all(a == b for a, b in zip(ranked_support, list(range(n_qubits)))):
        return operator
    batch_size = operator.size(-1)
    if not diagonal:
        operator = operator.view([2] * 2 * n_qubits + [batch_size])
        perm = list(
            tuple(ranked_support) + tuple(ranked_support + n_qubits) + (2 * n_qubits,)
        )
        if inv:
            perm = np.argsort(perm).tolist()
        return operator.permute(perm).reshape([2**n_qubits, 2**n_qubits, batch_size])
    else:
        operator = operator.view([2] * n_qubits + [batch_size])
        perm = list(tuple(ranked_support) + (n_qubits,))
        if inv:
            perm = np.argsort(perm).tolist()
        return operator.permute(perm).reshape([2**n_qubits, batch_size])


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


def dm_partial_trace(rho: DensityMatrix, keep_indices: list[int]) -> DensityMatrix:
    """
    Computes the partial trace of a density matrix for a system of several qubits with batch size.
    This function also permutes the qubits according to the order specified in keep_indices.

    Args:
        rho (DensityMatrix) : Density matrix of shape [2**n_qubits, 2**n_qubits, batch_size].
        keep_indices (list[int]): Index of the qubit subsystems to keep.

    Returns:
        DensityMatrix: Reduced density matrix after the partial trace,
        of shape [2**n_keep, 2**n_keep, batch_size].
    """
    n_qubits = int(log2((rho.shape[0])))
    batch_size = rho.shape[2]
    rho = rho.reshape(([2] * n_qubits * 2 + [batch_size]))

    rho_subscripts = array(ABC_ARRAY[: n_qubits * 2 + 1], copy=True)
    keep_subscripts = "".join(rho_subscripts[keep_indices]) + "".join(
        rho_subscripts[array(keep_indices) + n_qubits]
    )

    # Trace by equating indices
    trace_indices = delete(arange(n_qubits), keep_indices)
    rho_subscripts[trace_indices + n_qubits] = rho_subscripts[trace_indices]
    rho_subscripts = "".join(rho_subscripts)

    einsum_subscripts = (
        rho_subscripts + "->" + keep_subscripts + rho_subscripts[n_qubits * 2]
    )

    rho_reduced = torch.einsum(einsum_subscripts, rho)
    n_keep = len(keep_indices)
    return rho_reduced.reshape(2**n_keep, 2**n_keep, batch_size)


def generate_dm(n_qubits: int, batch_size: int) -> Tensor:
    """Generates a random density matrix using a real hermitian matrix"""
    density_list = []
    for _ in range(batch_size):
        rand_mat = torch.rand(2**n_qubits, 2**n_qubits) + 1j * torch.rand(
            2**n_qubits, 2**n_qubits
        )
        herm_mat = rand_mat + torch.transpose(rand_mat, 0, 1)
        density_mat = torch.matrix_exp(-herm_mat)
        density_mat = density_mat / torch.trace(density_mat)
        density_list.append(density_mat)

    return moveaxis(torch.stack(density_list), 0, 2)


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


def is_parametric(operation: pyq.Sequence) -> bool:
    """Check if operation is parametric.

    Args:
        operation (pyq.Sequence): checked operation

    Returns:
        bool: True if operation is parametric, False otherwise
    """

    from pyqtorch.primitives import Parametric

    res = False
    for m in operation.modules():
        if isinstance(m, (pyq.Scale, Parametric)):
            if isinstance(m.param_name, (str, pyq.ConcretizedCallable)):  # type: ignore[has-type]
                res = True
                break
    return res


def heaviside(x: Tensor, _: Any = None, slope: float = 1000.0) -> Tensor:
    """Torch autograd-compatible Heaviside function implementation.

    Args:
        x (Tensor): function argument
        _ (Any): unused argument left for signature compatibility reasons
        slope (float, optional): slope of Heaviside function (theoretically should be $infty$).
                                 Defaults to 1000.0.

    Returns:
        Tensor: function value
    """

    if x.ndim > 1:
        raise ValueError("Argument tensor must be 0-d or 1-d.")

    shape = (1, 2) if x.ndim == 0 else (len(x), 2)
    a = torch.zeros(shape)
    a[:, 0] = x
    return torch.clamp(
        slope * torch.max(a, dim=1)[0], torch.tensor(0.0), torch.tensor(1.0)
    )


def todense_tensor(x: Tensor) -> Tensor:
    """Convert a diagonal tensor to its dense representation.

    Args:
        x (Tensor): diagonal tensor

    Returns:
        Tensor: dense tensor
    """
    return torch.transpose(torch.diag_embed(torch.transpose(x, 0, 1)), 0, -1)
