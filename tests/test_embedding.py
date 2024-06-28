from __future__ import annotations

import torch

import pyqtorch as pyq


def test_sample_run_expectation_with_embedding() -> None:

    name0, fn0 = "fn0", pyq.embed.torch_call("sin", ["x"])
    name1, fn1 = "fn1", pyq.embed.torch_call("mul", ["x", "y"])
    embedding = pyq.Embedding(
        vparam_names=["x"], fparam_names=["y"], leaf_to_call={name0: fn0, name1: fn1}
    )
    rx = pyq.RX(0, param_name="fn0")
    cry = pyq.CRY(0, 1, param_name="fn1")
    cnot = pyq.CNOT(1, 2)
    ops = [rx, cry, cnot]
    n_qubits = 3
    circ = pyq.QuantumCircuit(n_qubits, ops)
    obs = pyq.Observable(n_qubits, [pyq.Z(0)])

    state = pyq.zero_state(n_qubits)

    theta_0_ad = torch.rand(1, requires_grad=True)
    theta_1_ad = torch.rand(1, requires_grad=True)

    values_ad = {"x": theta_0_ad, "y": theta_1_ad}
    wf = pyq.run(circ, state, values_ad, embedding)
    exp_ad = pyq.expectation(circ, state, values_ad, obs, pyq.DiffMode.AD, embedding)
    samples = pyq.sample(circ, state, values_ad, 100, embedding)
