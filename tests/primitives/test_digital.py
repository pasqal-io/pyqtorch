from __future__ import annotations

import random
from math import log2
from typing import Callable, Tuple

import pytest
import torch
from torch import Tensor

import pyqtorch as pyq
from pyqtorch import ConcretizedCallable
from pyqtorch.apply import apply_operator
from pyqtorch.matrices import (
    DEFAULT_MATRIX_DTYPE,
)
from pyqtorch.primitives import Parametric, Primitive
from pyqtorch.utils import (
    ATOL,
    product_state,
)

state_000 = product_state("000")
state_001 = product_state("001")
state_100 = product_state("100")
state_101 = product_state("101")
state_110 = product_state("110")
state_011 = product_state("011")
state_111 = product_state("111")
state_0000 = product_state("0000")
state_1110 = product_state("1110")
state_1111 = product_state("1111")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_identity() -> None:
    assert torch.allclose(product_state("0"), pyq.I(0)(product_state("0"), None))
    assert torch.allclose(product_state("1"), pyq.I(0)(product_state("1")))


def test_N() -> None:
    null_state = torch.zeros_like(pyq.zero_state(1))
    assert torch.allclose(null_state, pyq.N(0)(product_state("0"), None))
    assert torch.allclose(product_state("1"), pyq.N(0)(product_state("1"), None))


@pytest.mark.parametrize("n_qubits", [1, 2, 3])
def test_projectors(n_qubits: int) -> None:
    qubit_support = tuple(range(n_qubits))
    max_possibility = 2**n_qubits
    possible_nbs = list(range(max_possibility - 1))
    for ket in [random.choice(possible_nbs) for _ in range(2 * n_qubits)]:
        for bra in [random.choice(possible_nbs) for _ in range(2 * n_qubits)]:
            t_mat = torch.zeros(max_possibility, dtype=DEFAULT_MATRIX_DTYPE)
            t_mat[ket] = 1.0 + 0.0j
            t_mat = t_mat.reshape((2,) * n_qubits + (1,))
            assert torch.allclose(
                pyq.Projector(
                    qubit_support, ket=f"{ket:0{n_qubits}b}", bra=f"{bra:0{n_qubits}b}"
                )(product_state(f"{bra:0{n_qubits}b}")),
                t_mat,
            )
        if ket > 0:
            assert torch.allclose(
                pyq.Projector(
                    qubit_support, ket=f"{ket:0{n_qubits}b}", bra=f"{ket:0{n_qubits}b}"
                )(product_state(f"{ket-1:0{n_qubits}b}")),
                torch.zeros((2,) * n_qubits + (1,), dtype=DEFAULT_MATRIX_DTYPE),
            )


czop_example = pyq.CZ(control=(0, 1), target=2)
crxop_example = pyq.CRX(control=(0, 1), target=2, param_name="theta")


@pytest.mark.parametrize(
    "projector, initial_state, final_state",
    [
        (czop_example, state_101, state_101),
        (czop_example, state_111, -state_111),
        (crxop_example, state_001, state_001),
        (
            crxop_example,
            state_110,
            torch.tensor(
                [
                    [
                        [[0.0000 + 0.0000j], [0.0000 + 0.0000j]],
                        [[0.0000 + 0.0000j], [0.0000 + 0.0000j]],
                    ],
                    [
                        [[0.0000 + 0.0000j], [0.0000 + 0.0000j]],
                        [[0.8776 + 0.0000j], [0.0000 - 0.4794j]],
                    ],
                ],
                dtype=torch.complex128,
            ),
        ),
        (
            crxop_example,
            state_111,
            torch.tensor(
                [
                    [
                        [[0.0000 + 0.0000j], [0.0000 + 0.0000j]],
                        [[0.0000 + 0.0000j], [0.0000 + 0.0000j]],
                    ],
                    [
                        [[0.0000 + 0.0000j], [0.0000 + 0.0000j]],
                        [[0.0000 - 0.4794j], [0.8776 + 0.0000j]],
                    ],
                ],
                dtype=torch.complex128,
            ),
        ),
    ],
)
def test_multicontrol_rotation(
    projector: Primitive, initial_state: torch.Tensor, final_state: torch.Tensor
) -> None:

    val_param = {"theta": torch.Tensor([1.0])}
    projector_apply_res = projector(initial_state, val_param)
    assert torch.allclose(final_state, projector_apply_res, atol=1.0e-4)


def test_CNOT_state00_controlqubit_0() -> None:
    result: Tensor = pyq.CNOT(0, 1)(product_state("00"), None)
    assert torch.equal(product_state("00"), result)


def test_CNOT_state10_controlqubit_0() -> None:
    result: Tensor = pyq.CNOT(0, 1)(product_state("10"), None)
    assert torch.allclose(product_state("11"), result)


def test_CNOT_state11_controlqubit_0() -> None:
    result: Tensor = pyq.CNOT(0, 1)(product_state("11"), None)
    assert torch.allclose(product_state("10"), result)


def test_CRY_state10_controlqubit_0() -> None:
    result: Tensor = pyq.CRY(0, 1, "theta")(
        product_state("10"), {"theta": torch.tensor([torch.pi])}
    )
    assert torch.allclose(product_state("11"), result, atol=ATOL)


def test_CRY_state01_controlqubit_0() -> None:
    result: Tensor = pyq.CRY(1, 0, "theta")(
        product_state("01"), {"theta": torch.tensor([torch.pi])}
    )
    assert torch.allclose(product_state("11"), result, atol=ATOL)


@pytest.mark.parametrize(
    "initial_state,expected_state",
    [
        (state_000, state_000),
        (state_001, state_001),
        (state_100, state_100),
        (state_101, state_110),
        (state_110, state_101),
    ],
)
def test_CSWAP_controlqubits0(initial_state: Tensor, expected_state: Tensor) -> None:
    cswap = pyq.CSWAP(0, (1, 2))
    assert torch.allclose(cswap(initial_state, None), expected_state)


@pytest.mark.parametrize(
    "initial_state,expected_state",
    [
        (state_000, state_000),
        (state_001, state_001),
        (state_100, state_100),
        (state_101, state_101),
        (state_110, state_111),
        (state_1110, state_1111),
    ],
)
def test_Toffoli_controlqubits0(initial_state: Tensor, expected_state: Tensor) -> None:
    n_qubits = int(log2(torch.numel(initial_state)))
    qubits = tuple([i for i in range(n_qubits)])
    toffoli = pyq.Toffoli(qubits[:-1], qubits[-1])
    assert torch.allclose(toffoli(initial_state, None), expected_state)


@pytest.mark.parametrize(
    "initial_state,expects_rotation",
    [
        (state_000, False),
        (state_001, False),
        (state_100, False),
        (state_101, False),
        (state_110, True),
        (state_1110, True),
    ],
)
@pytest.mark.parametrize("gate", ["RX", "RY", "RZ", "PHASE"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_multi_controlled_gates(
    initial_state: Tensor,
    expects_rotation: bool,
    batch_size: int,
    gate: str,
    dtype: torch.dtype,
) -> None:
    phi = "phi"

    initial_state = initial_state.to(device=device, dtype=dtype)
    rot_gate = getattr(pyq, gate)
    controlled_rot_gate = getattr(pyq, "C" + gate)
    phi = torch.rand(batch_size).to(device=device)
    n_qubits = int(log2(torch.numel(initial_state)))
    qubits = tuple([i for i in range(n_qubits)])
    op = controlled_rot_gate(qubits[:-1], qubits[-1], "phi").to(
        device=device, dtype=dtype
    )
    out = op(initial_state, {"phi": phi})
    expected_state = (
        rot_gate(qubits[-1], "phi").to(device=device, dtype=dtype)(
            initial_state, {"phi": phi}
        )
        if expects_rotation
        else initial_state
    )
    assert torch.allclose(out, expected_state)
    if gate != "PHASE":
        assert len(op.spectral_gap) == 2
    else:
        assert op.spectral_gap == 2.0

    assert op.eigenvals_generator.dtype == dtype


@pytest.mark.parametrize(
    "state_fn", [pyq.random_state, pyq.zero_state, pyq.uniform_state]
)
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("n_qubits", [1, 2, 4])
def test_parametrized_phase_gate(
    state_fn: Callable, batch_size: int, n_qubits: int
) -> None:
    target: int = torch.randint(low=0, high=n_qubits, size=(1,)).item()
    state = state_fn(n_qubits, batch_size=batch_size)
    phi = torch.tensor([torch.pi / 2], dtype=DEFAULT_MATRIX_DTYPE)
    phase = pyq.PHASE(target, "phi")
    constant_phase = pyq.S(target)
    assert torch.allclose(phase(state, {"phi": phi}), constant_phase(state, None))
    assert phase.spectral_gap == 2.0
    assert constant_phase.spectral_gap == 1.0


def test_dagger_single_qubit() -> None:
    for cls in [
        pyq.X,
        pyq.Y,
        pyq.Z,
        pyq.S,
        pyq.H,
        pyq.T,
        pyq.RX,
        pyq.RY,
        pyq.RZ,
        pyq.PHASE,
    ]:
        n_qubits = torch.randint(low=1, high=4, size=(1,)).item()
        target = random.choice([i for i in range(n_qubits)])
        state = pyq.random_state(n_qubits)
        for param_name in ["theta", ""]:
            if issubclass(cls, Parametric):
                op = cls(target, param_name)  # type: ignore[arg-type]
            else:
                op = cls(target)  # type: ignore[assignment, call-arg]
            values = (
                {param_name: torch.rand(1)} if param_name == "theta" else torch.rand(1)
            )
            new_state = apply_operator(state, op.tensor(values), [target])
            daggered_back = apply_operator(new_state, op.dagger(values), [target])
            assert torch.allclose(daggered_back, state)


def test_dagger_nqubit() -> None:
    for cls in [
        pyq.SWAP,
        pyq.CNOT,
        pyq.CY,
        pyq.CZ,
        pyq.CRX,
        pyq.CRY,
        pyq.CRZ,
        pyq.CPHASE,
    ]:
        qubit_support: Tuple[int, ...]
        n_qubits = torch.randint(low=3, high=8, size=(1,)).item()
        target = random.choice([i for i in range(n_qubits - 2)])
        state = pyq.random_state(n_qubits)
        for param_name in ["theta", ""]:
            if isinstance(cls, (pyq.CSWAP, pyq.Toffoli)):
                op = cls((target - 2, target - 1), target)
                qubit_support = (target + 2, target + 1, target)
            elif issubclass(cls, Parametric):
                op = cls(target - 1, target, param_name)  # type: ignore[arg-type]
                qubit_support = (target + 1, target)
            else:
                op = cls(target - 1, target)  # type: ignore[call-arg]
                qubit_support = (target + 1, target)
            values = (
                {param_name: torch.rand(1)} if param_name == "theta" else torch.rand(1)
            )
            new_state = apply_operator(state, op.tensor(values), qubit_support)
            daggered_back = apply_operator(new_state, op.dagger(values), qubit_support)
            assert torch.allclose(daggered_back, state)


def test_U() -> None:
    n_qubits = torch.randint(low=1, high=8, size=(1,)).item()
    target = random.choice([i for i in range(n_qubits)])
    params = ["phi", "theta", "omega"]
    u = pyq.U(target, phi=params[0], theta=params[1], omega=params[2])
    values = {param: torch.rand(1) for param in params}
    state = pyq.random_state(n_qubits)
    assert torch.allclose(
        u(state, values),
        pyq.QuantumCircuit(n_qubits, u.digital_decomposition())(state, values),
    )
    assert u.spectral_gap == 2.0


@pytest.mark.parametrize("gate", [pyq.RX, pyq.RY, pyq.RZ])
def test_parametric_constantparam(gate: Parametric) -> None:
    n_qubits = 4
    max_batch_size = 10
    target = torch.randint(0, n_qubits, (1,)).item()
    param_val = torch.rand(torch.randint(1, max_batch_size, (1,)).item())
    state = pyq.random_state(n_qubits)
    assert torch.allclose(
        gate(target, "theta")(state, {"theta": param_val}),
        gate(target, param_val)(state),
    )


@pytest.mark.parametrize("gate", [pyq.RX, pyq.RY, pyq.RZ])
def test_parametric_callableparam(gate: Parametric) -> None:
    n_qubits = 4
    max_batch_size = 10
    target = torch.randint(0, n_qubits, (1,)).item()
    size = torch.randint(1, max_batch_size, (1,)).item()
    param_val_x = torch.rand(size)
    param_val_y = torch.rand(size)
    state = pyq.random_state(n_qubits)
    param = ConcretizedCallable("add", ["x", "y"])
    assert torch.allclose(
        gate(target, param)(state, {"x": param_val_x, "y": param_val_y}),
        gate(target, param_val_x + param_val_y)(state),
    )
