## Time-dependent Schrödinger and Lindblad master equation solving

For simulating systems described by time-dependent Hamiltonians `pyqtorch` has a module implementing Schrödinger and Lindblad equation solvers.

```python exec="on" source="material-block"
import torch
from torch import Tensor
from pyqtorch.utils import product_state, operator_kron
from pyqtorch.matrices import XMAT, YMAT, IMAT
from pyqtorch.time_dependent.mesolve import mesolve
from pyqtorch.time_dependent.sesolve import sesolve
from pyqtorch.utils import SolverType

n_qubits = 2
duration = 1.0  # simulation duration
n_steps = 1000  # number of solver time steps

# prepare initial state
input_state = product_state("0"*n_qubits).reshape((-1, 1))

# prepare time-dependent Hamiltonian function
def ham_t(t: float) -> Tensor:
    t = torch.as_tensor(t)
    return 10.0 * (
        2.0 * torch.sin(t) * torch.kron(XMAT, torch.eye(2))
        + 3.0 * t**2 * torch.kron(torch.eye(2), YMAT)
    )

# solve Schrodinger equation for the system
t_points = torch.linspace(0, duration, n_steps)
final_state_se = sesolve(ham_t, input_state, t_points, SolverType.DP5_SE).states[-1]

# define jump operator L
L = IMAT.clone()
for i in range(n_qubits-1):
    L = torch.kron(L, XMAT)

# prepare initial density matrix with batch dimension as the last
rho0 = torch.matmul(input_state, input_state.T).unsqueeze(-1)

# solve Lindblad master equation
final_state_me = mesolve(ham_t, rho0, [L], t_points, SolverType.DP5_ME).states[-1]

```
