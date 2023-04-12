import time

import torch
from pyqtorch.modules import RX, X, CNOT, QuantumCircuit, uniform_state
from pyqtorch.core.batched_operation import batchedRX

# print(OPERATIONS_DICT["I"])
# g = OPERATIONS_DICT["I"].to(device="mps", dtype=torch.cfloat)
# print(g + g)

dtype = torch.cdouble
device = "cpu"
batch_size = 1_000
state = uniform_state(2, batch_size=batch_size, device=device, dtype=dtype)
phi = torch.rand(batch_size, device=device, dtype=dtype)
thetas = {"phi": phi}

qubits = [0]
n_qubits = 2
iters = 1_000

t1 = time.time()
for _ in range(iters):
    batchedRX(phi, state, qubits, n_qubits)
print(f"{time.time()-t1}")

gate = RX(qubits, n_qubits, "phi").to(device=device, dtype=dtype)
# gate = torch.compile(gate)
t1 = time.time()
for _ in range(iters):
    gate(thetas, state)
print(f"{time.time()-t1}")
print(state)

ops = [X([0], 2), X([1], 2), RX([1], 2, "phi"), CNOT([0, 1], 2)]
ops = [RX([0], 2, "phi")]
circ = QuantumCircuit(2, ops).to(device=device, dtype=dtype)

circ(thetas, state)
t1 = time.time()
for _ in range(iters):
    circ(thetas, state)
print(f"{time.time()-t1}")
print(state)

print()
