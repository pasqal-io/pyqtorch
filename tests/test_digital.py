from __future__ import annotations

import random
from math import log2
from typing import Callable, Tuple

import pytest
import torch
from conftest import _calc_mat_vec_wavefunction
from torch import Tensor

import pyqtorch as pyq
from pyqtorch.apply import apply_operator, operator_product
from pyqtorch.matrices import (
    DEFAULT_MATRIX_DTYPE,
    HMAT,
    IMAT,
    XMAT,
    YMAT,
    ZMAT,
    _dagger,
)
from pyqtorch.noise import (
    AmplitudeDamping,
    GeneralizedAmplitudeDamping,
    Noise,
    PhaseDamping,
)
from pyqtorch.parametric import Parametric
from pyqtorch.primitive import H, I, Primitive, S, T, X, Y, Z
from pyqtorch.utils import (
    ATOL,
    RTOL,
    DensityMatrix,
    density_mat,
    operator_kron,
    product_state,
    promote_operator,
    random_state,
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


def test_identity() -> None:
    assert torch.allclose(product_state("0"), pyq.I(0)(product_state("0"), None))
    assert torch.allclose(product_state("1"), pyq.I(1)(product_state("1")))


def test_N() -> None:
    null_state = torch.zeros_like(pyq.zero_state(1))
    assert torch.allclose(null_state, pyq.N(0)(product_state("0"), None))
    assert torch.allclose(product_state("1"), pyq.N(0)(product_state("1"), None))


@pytest.mark.parametrize("gate", [I, X, Y, Z, H, T, S])
@pytest.mark.parametrize("n_qubits", [1, 2, 4])
def test_single_qubit_gates(gate: Primitive, n_qubits: int) -> None:
    target = torch.randint(0, n_qubits, (1,)).item()
    block = gate(target)
    init_state = pyq.random_state(n_qubits)
    wf_pyq = block(init_state, None)
    wf_mat = _calc_mat_vec_wavefunction(block, n_qubits, init_state)
    assert torch.allclose(wf_mat, wf_pyq, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("batch_size", [i for i in range(2, 10)])
@pytest.mark.parametrize("gate", [pyq.RX, pyq.RY, pyq.RZ])
@pytest.mark.parametrize("n_qubits", [1, 2, 4])
def test_rotation_gates(batch_size: int, gate: Primitive, n_qubits: int) -> None:
    params = [f"th{i}" for i in range(gate.n_params)]
    values = {param: torch.rand(batch_size) for param in params}
    target = torch.randint(0, n_qubits, (1,)).item()

    init_state = pyq.random_state(n_qubits)
    block = gate(target, *params)
    wf_pyq = block(init_state, values)
    wf_mat = _calc_mat_vec_wavefunction(block, n_qubits, init_state, values=values)
    assert torch.allclose(wf_mat, wf_pyq, rtol=RTOL, atol=ATOL)


def test_projectors() -> None:
    t0 = torch.tensor([[0.0], [0.0]], dtype=DEFAULT_MATRIX_DTYPE)
    t1 = torch.tensor([[1.0], [0.0]], dtype=DEFAULT_MATRIX_DTYPE)
    t2 = torch.tensor([[0.0], [1.0]], dtype=DEFAULT_MATRIX_DTYPE)
    assert torch.allclose(t1, pyq.Projector(0, ket="0", bra="0")(product_state("0")))
    assert torch.allclose(t0, pyq.Projector(0, ket="0", bra="0")(product_state("1")))
    assert torch.allclose(t2, pyq.Projector(0, ket="1", bra="1")(product_state("1")))
    assert torch.allclose(t0, pyq.Projector(0, ket="1", bra="1")(product_state("0")))
    t00 = torch.tensor(
        [[[1.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]],
        dtype=torch.complex128,
    )
    t01 = torch.tensor(
        [[[0.0 + 0.0j], [1.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]],
        dtype=torch.complex128,
    )
    t10 = torch.tensor(
        [[[0.0 + 0.0j], [0.0 + 0.0j]], [[1.0 + 0.0j], [0.0 + 0.0j]]],
        dtype=torch.complex128,
    )
    t11 = torch.tensor(
        [[[0.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [1.0 + 0.0j]]],
        dtype=torch.complex128,
    )
    assert torch.allclose(
        pyq.Projector((0, 1), ket="00", bra="00")(product_state("00")), t00
    )
    assert torch.allclose(
        pyq.Projector((0, 1), ket="10", bra="01")(product_state("01")), t10
    )
    assert torch.allclose(
        pyq.Projector((0, 1), ket="01", bra="10")(product_state("10")), t01
    )
    assert torch.allclose(
        pyq.Projector((0, 1), ket="11", bra="11")(product_state("11")), t11
    )
    t000 = torch.tensor(
        [
            [[[1.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]],
            [[[0.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]],
        ],
        dtype=torch.complex128,
    )
    t100 = torch.tensor(
        [
            [[[0.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]],
            [[[1.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]],
        ],
        dtype=torch.complex128,
    )
    t001 = torch.tensor(
        [
            [[[0.0 + 0.0j], [1.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]],
            [[[0.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]],
        ],
        dtype=torch.complex128,
    )
    t010 = torch.tensor(
        [
            [[[0.0 + 0.0j], [0.0 + 0.0j]], [[1.0 + 0.0j], [0.0 + 0.0j]]],
            [[[0.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]],
        ],
        dtype=torch.complex128,
    )
    t111 = torch.tensor(
        [
            [[[0.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]],
            [[[0.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [1.0 + 0.0j]]],
        ],
        dtype=torch.complex128,
    )
    assert torch.allclose(
        pyq.Projector((0, 1, 2), ket="000", bra="000")(product_state("000")), t000
    )
    assert torch.allclose(
        pyq.Projector((0, 1, 2), ket="100", bra="001")(product_state("001")), t100
    )
    assert torch.allclose(
        pyq.Projector((0, 1, 2), ket="010", bra="010")(product_state("010")), t010
    )
    assert torch.allclose(
        pyq.Projector((0, 1, 2), ket="001", bra="100")(product_state("100")), t001
    )
    assert torch.allclose(
        pyq.Projector((0, 1, 2), ket="111", bra="111")(product_state("111")), t111
    )


@pytest.mark.parametrize(
    "projector, exp_projector_mat",
    [
        (
            pyq.Projector(0, bra="1", ket="1"),
            torch.tensor(
                [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
                dtype=torch.complex128,
            ),
        ),
        (
            pyq.N(0),
            (IMAT - ZMAT) / 2.0,
        ),
        (
            pyq.CNOT(0, 1),
            torch.tensor(
                [
                    [
                        [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                    ]
                ],
                dtype=torch.complex128,
            ),
        ),
    ],
)
def test_projector_tensor(
    projector: Primitive, exp_projector_mat: torch.Tensor
) -> None:

    nbqubits = int(log2(exp_projector_mat.shape[-1]))
    projector_mat = projector.tensor(
        n_qubits=nbqubits, values={"theta": torch.Tensor([1.0])}
    ).squeeze(-1)
    assert torch.allclose(projector_mat, exp_projector_mat, atol=1.0e-4)


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
    print(final_state, projector_apply_res)

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
def test_multi_controlled_gates(
    initial_state: Tensor, expects_rotation: bool, batch_size: int, gate: str
) -> None:
    phi = "phi"
    rot_gate = getattr(pyq, gate)
    controlled_rot_gate = getattr(pyq, "C" + gate)
    phi = torch.rand(batch_size)
    n_qubits = int(log2(torch.numel(initial_state)))
    qubits = tuple([i for i in range(n_qubits)])
    op = controlled_rot_gate(qubits[:-1], qubits[-1], "phi")
    out = op(initial_state, {"phi": phi})
    expected_state = (
        rot_gate(qubits[-1], "phi")(initial_state, {"phi": phi})
        if expects_rotation
        else initial_state
    )
    assert torch.allclose(out, expected_state)


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
            new_state = apply_operator(state, op.unitary(values), [target])
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
            new_state = apply_operator(state, op.unitary(values), qubit_support)
            daggered_back = apply_operator(new_state, op.dagger(values), qubit_support)
            assert torch.allclose(daggered_back, state)


def test_U() -> None:
    n_qubits = torch.randint(low=1, high=8, size=(1,)).item()
    target = random.choice([i for i in range(n_qubits)])
    params = ["phi", "theta", "omega"]
    u = pyq.U(target, *params)
    values = {param: torch.rand(1) for param in params}
    state = pyq.random_state(n_qubits)
    assert torch.allclose(
        u(state, values),
        pyq.QuantumCircuit(n_qubits, u.digital_decomposition())(state, values),
    )


def test_dm(n_qubits: int, batch_size: int) -> None:
    state = random_state(n_qubits)
    projector = torch.outer(state.flatten(), state.conj().flatten()).view(
        2**n_qubits, 2**n_qubits, 1
    )
    dm = density_mat(state)
    assert dm.size() == torch.Size([2**n_qubits, 2**n_qubits, 1])
    assert torch.allclose(dm, projector)
    assert torch.allclose(dm.squeeze(), dm.squeeze() @ dm.squeeze())
    states = []
    projectors = []
    for batch in range(batch_size):
        state = random_state(n_qubits)
        states.append(state)
        projector = torch.outer(state.flatten(), state.conj().flatten()).view(
            2**n_qubits, 2**n_qubits, 1
        )
        projectors.append(projector)
    dm_proj = torch.cat(projectors, dim=2)
    state_cat = torch.cat(states, dim=n_qubits)
    dm = density_mat(state_cat)
    assert dm.size() == torch.Size([2**n_qubits, 2**n_qubits, batch_size])
    assert torch.allclose(dm, dm_proj)


def test_promote(random_gate: Primitive, n_qubits: int, target: int) -> None:
    op_prom = promote_operator(random_gate.unitary(), target, n_qubits)
    assert op_prom.size() == torch.Size([2**n_qubits, 2**n_qubits, 1])
    assert torch.allclose(
        operator_product(op_prom, _dagger(op_prom), target),
        torch.eye(2**n_qubits, dtype=torch.cdouble).unsqueeze(2),
    )


def test_operator_product(random_gate: Primitive, n_qubits: int, target: int) -> None:
    op = random_gate
    batch_size_1 = torch.randint(low=1, high=5, size=(1,)).item()
    batch_size_2 = torch.randint(low=1, high=5, size=(1,)).item()
    max_batch = max(batch_size_2, batch_size_1)
    op_prom = promote_operator(op.unitary(), target, n_qubits).repeat(
        1, 1, batch_size_1
    )
    op_mul = operator_product(
        op.unitary().repeat(1, 1, batch_size_2), _dagger(op_prom), target
    )
    assert op_mul.size() == torch.Size([2**n_qubits, 2**n_qubits, max_batch])
    assert torch.allclose(
        op_mul,
        torch.eye(2**n_qubits, dtype=torch.cdouble)
        .unsqueeze(2)
        .repeat(1, 1, max_batch),
    )


@pytest.mark.parametrize(
    "operator,matrix", [(I, IMAT), (X, XMAT), (Z, ZMAT), (Y, YMAT), (H, HMAT)]
)
def test_operator_kron(operator: Tensor, matrix: Tensor) -> None:
    n_qubits = torch.randint(low=1, high=5, size=(1,)).item()
    batch_size = torch.randint(low=1, high=5, size=(1,)).item()
    states, krons = [], []
    for batch in range(batch_size):
        state = random_state(n_qubits)
        states.append(state)
        kron = torch.kron(density_mat(state).squeeze(), matrix).unsqueeze(2)
        krons.append(kron)
    input_state = torch.cat(states, dim=n_qubits)
    kron_out = operator_kron(density_mat(input_state), operator(0).dagger())
    assert kron_out.size() == torch.Size(
        [2 ** (n_qubits + 1), 2 ** (n_qubits + 1), batch_size]
    )
    kron_expect = torch.cat(krons, dim=2)
    assert torch.allclose(kron_out, kron_expect)
    assert torch.allclose(
        torch.kron(operator(0).dagger().contiguous(), I(0).unitary()),
        operator_kron(operator(0).dagger(), I(0).unitary()),
    )


def test_kron_batch() -> None:
    n_qubits = torch.randint(low=1, high=5, size=(1,)).item()
    batch_size_1 = torch.randint(low=1, high=5, size=(1,)).item()
    batch_size_2 = torch.randint(low=1, high=5, size=(1,)).item()
    max_batch = max(batch_size_1, batch_size_2)
    dm_1 = density_mat(random_state(n_qubits, batch_size_1))
    dm_2 = density_mat(random_state(n_qubits, batch_size_2))
    dm_out = operator_kron(dm_1, dm_2)
    assert dm_out.size() == torch.Size(
        [2 ** (2 * n_qubits), 2 ** (2 * n_qubits), max_batch]
    )
    if batch_size_1 > batch_size_2:
        dm_2 = dm_2.repeat(1, 1, batch_size_1)[:, :, :batch_size_1]
    elif batch_size_2 > batch_size_1:
        dm_1 = dm_1.repeat(1, 1, batch_size_2)[:, :, :batch_size_2]
    density_matrices = []
    for batch in range(max_batch):
        density_matrice = torch.kron(dm_1[:, :, batch], dm_2[:, :, batch]).unsqueeze(2)
        density_matrices.append(density_matrice)
    dm_expect = torch.cat(density_matrices, dim=2)
    assert torch.allclose(dm_out, dm_expect)


def test_circuit_tensor() -> None:
    ops = [pyq.RX(0, "theta_0"), pyq.RY(0, "theta_1"), pyq.RX(1, "theta_2")]
    circ = pyq.QuantumCircuit(2, ops)
    values = {f"theta_{i}": torch.Tensor([float(i)]) for i in range(3)}
    tensorcirc = circ.tensor(values)
    assert tensorcirc.size() == (4, 4, 1)
    assert torch.allclose(
        tensorcirc,
        torch.tensor(
            [
                [
                    [0.4742 + 0.0000j],
                    [0.0000 - 0.7385j],
                    [-0.2590 + 0.0000j],
                    [0.0000 + 0.4034j],
                ],
                [
                    [0.0000 - 0.7385j],
                    [0.4742 + 0.0000j],
                    [0.0000 + 0.4034j],
                    [-0.2590 + 0.0000j],
                ],
                [
                    [0.2590 + 0.0000j],
                    [0.0000 - 0.4034j],
                    [0.4742 + 0.0000j],
                    [0.0000 - 0.7385j],
                ],
                [
                    [0.0000 - 0.4034j],
                    [0.2590 + 0.0000j],
                    [0.0000 - 0.7385j],
                    [0.4742 + 0.0000j],
                ],
            ],
            dtype=torch.complex128,
        ),
        atol=1.0e-4,
    )


@pytest.mark.parametrize("n_qubits", [{"low": 2, "high": 5}], indirect=True)
def test_flip_gates(
    n_qubits: int,
    target: int,
    batch_size: int,
    rho_input: Tensor,
    random_flip_gate: Noise,
    flip_expected_state: DensityMatrix,
    flip_probability: Tensor | float,
    flip_gates_prob_0: Noise,
    flip_gates_prob_1: tuple,
    random_input_state: Tensor,
) -> None:
    FlipGate = random_flip_gate
    output_state: DensityMatrix = FlipGate(target, flip_probability)(rho_input)
    assert output_state.size() == torch.Size([2**n_qubits, 2**n_qubits, batch_size])
    assert torch.allclose(output_state, flip_expected_state)

    input_state = random_input_state  # fix the same random state for every call
    assert torch.allclose(
        flip_gates_prob_0(density_mat(input_state)), density_mat(input_state)
    )

    FlipGate_1, expected_op = flip_gates_prob_1
    assert torch.allclose(FlipGate_1(density_mat(input_state)), expected_op)


def test_damping_gates(
    n_qubits: int,
    target: int,
    batch_size: int,
    random_damping_gate: Noise,
    damping_expected_state: tuple,
    damping_rate: Tensor,
    damping_gates_prob_0: Tensor,
    random_input_state: Tensor,
    rho_input: Tensor,
) -> None:
    DampingGate, expected_state = damping_expected_state
    apply_gate = DampingGate(rho_input)
    assert apply_gate.size() == torch.Size([2**n_qubits, 2**n_qubits, batch_size])
    assert torch.allclose(apply_gate, expected_state)

    input_state = random_input_state
    assert torch.allclose(
        damping_gates_prob_0(input_state), density_mat(I(target)(input_state))
    )

    rho_0: DensityMatrix = density_mat(product_state("0", batch_size))
    rho_1: DensityMatrix = density_mat(product_state("1", batch_size))
    if DampingGate == AmplitudeDamping:
        assert torch.allclose(DampingGate(target, rate=1)(rho_1), rho_0)
    elif DampingGate == PhaseDamping:
        assert torch.allclose(DampingGate(target, rate=1)(rho_1), I(target)(rho_1))
    elif DampingGate == GeneralizedAmplitudeDamping:
        assert torch.allclose(DampingGate(target, probability=1, rate=1)(rho_1), rho_0)
