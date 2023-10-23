from __future__ import annotations

import pytest
import torch

import pyqtorch as pyq


def get_state(bitstring: str) -> torch.Tensor:
    state = torch.zeros(2 ** len(bitstring), dtype=torch.complex128)
    state[int(bitstring, 2)] = torch.tensor(1.0 + 0j, dtype=torch.complex128)
    return state.reshape([2] * len(bitstring) + [1])


def test_identity() -> None:
    assert torch.allclose(get_state("0"), pyq.I(0)(get_state("0"), None))
    assert torch.allclose(get_state("1"), pyq.I(1)(get_state("1")))


@pytest.mark.xfail
def test_N() -> None:
    assert torch.allclose(get_state("0"), pyq.N(0)(get_state("0"), None))
    assert torch.allclose(get_state("0"), pyq.N(0)(get_state("1"), None))


def test_CNOT_state00_controlqubit_0() -> None:
    result: torch.Tensor = pyq.CNOT(0, 1)(get_state("00"), None)
    assert torch.equal(get_state("00"), result)


def test_CNOT_state10_controlqubit_0() -> None:
    result: torch.Tensor = pyq.CNOT(0, 1)(get_state("10"), None)
    assert torch.equal(get_state("11"), result)


def test_CNOT_state11_controlqubit_0() -> None:
    result: torch.Tensor = pyq.CNOT(0, 1)(get_state("11"), None)
    assert torch.equal(get_state("10"), result)


def test_CRY_state10_controlqubit_0() -> None:
    result: torch.Tensor = pyq.CRY(0, 1, "theta")(
        get_state("10"), {"theta": torch.tensor([torch.pi])}
    )
    assert torch.allclose(get_state("11"), result)


def test_CRY_state01_controlqubit_0() -> None:
    result: torch.Tensor = pyq.CRY(1, 0, "theta")(
        get_state("01"), {"theta": torch.tensor([torch.pi])}
    )
    assert torch.allclose(get_state("11"), result)


def test_CSWAP_state101_controlqubit_0() -> None:
    result: torch.Tensor = pyq.CSWAP([0, 1], 2)(get_state("101"), None)
    assert torch.allclose(get_state("110"), result)


def test_CSWAP_state110_controlqubit_0() -> None:
    result: torch.Tensor = pyq.CSWAP([0, 1], 2)(get_state("101"), None)
    assert torch.allclose(get_state("110"), result)
