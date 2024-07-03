from __future__ import annotations

import pytest
import torch
import torch.autograd.gradcheck

import pyqtorch as pyq


@pytest.mark.parametrize("diff_mode", [pyq.DiffMode.AD])
def test_sample_run_expectation_grads_with_embedding(diff_mode) -> None:
    name0, fn0 = "fn0", pyq.embed.torch_call("sin", ["x"])
    name1, fn1 = "fn1", pyq.embed.torch_call("mul", ["fn0", "y"])
    name2, fn2 = "fn2", pyq.embed.torch_call("mul", ["fn1", 2.0])
    name3, fn3 = "fn3", pyq.embed.torch_call("log", ["fn2"])
    embedding = pyq.Embedding(
        vparam_names=["x"],
        fparam_names=["y"],
        leaf_to_call={name0: fn0, name1: fn1, name2: fn2, name3: fn3},
    )
    rx = pyq.RX(0, param_name=name0)
    cry = pyq.CRY(0, 1, param_name=name1)
    phase = pyq.PHASE(1, param_name=name2)
    ry = pyq.RY(1, param_name=name3)
    cnot = pyq.CNOT(1, 2)
    ops = [rx, cry, phase, ry, cnot]
    n_qubits = 3
    circ = pyq.QuantumCircuit(n_qubits, ops)
    obs = pyq.Observable(n_qubits, [pyq.Z(0)])

    state = pyq.zero_state(n_qubits)

    x = torch.rand(1, requires_grad=True)
    y = torch.rand(1, requires_grad=True)

    values_ad = {"x": x, "y": y}
    wf = pyq.run(circ, state, values_ad, embedding)
    samples = pyq.sample(circ, state, values_ad, 100, embedding)
    exp_ad = pyq.expectation(circ, state, values_ad, obs, diff_mode, embedding)
    assert torch.autograd.gradcheck(
        lambda x, y: pyq.expectation(
            circ, state, {"x": x, "y": y}, obs, diff_mode, embedding
        ),
        (x, y),
        atol=0.2,
    )
