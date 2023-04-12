import time

import torch
from pyqtorch.modules import RX, X, CNOT, QuantumCircuit, uniform_state, zero_state
from pyqtorch.core.batched_operation import batchedRX

def timeit(f, *args, niters=100):
    t = 0
    for _ in range(niters):
        t0 = time.time()
        f(*args)
        t1 = time.time()
        t += t1-t0
    return t / niters

dtype = torch.cdouble
device = "cuda"
batch_size = 1_000
qubits = [0]
n_qubits = 10

state = zero_state(n_qubits, batch_size=batch_size, device=device, dtype=dtype)
phi = torch.rand(batch_size, device=device, dtype=dtype)
thetas = {"phi": phi}

func_time = timeit(batchedRX, phi, state, qubits, n_qubits)

gate = RX(qubits, n_qubits, "phi").to(device=device, dtype=dtype)
mod_time = timeit(gate, thetas, state)

print(f"Functional pyq: {func_time}")
print(f"Module pyq:     {mod_time}")
