from __future__ import annotations

import torch
from matrix import custom_tensordot

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

A = torch.rand(2, 2, dtype=torch.complex128)
B = torch.rand(2, dtype=torch.complex128)

# for i in [10**n for n in range(2,7)]:
#     st = time.time()
#     for _ in range(i):
#         C = torch.matmul(A,B)
#     et = time.time()
#     print(f"torch.matmul {i} took {et - st}")

#     AA = CustomMatrix(A)
#     BB = CustomMatrix(B)

#     st = time.time()
#     for _ in range(i):
#         CC = AA @ BB
#     et = time.time()
#     print(f"CustomMatrix.matmul {i} took {et - st}")

#     # opt_matmul = torch.compile(custom_matmul)
#     # real_part, imag_part = opt_matmul(A.real, A.imag, B.real, B.imag)
#     # print(torch.complex(real_part, imag_part))

#     st = time.time()
#     for _ in range(i):
#         real_part, imag_part = custom_matmul(A.real, A.imag, B.real, B.imag)
#     et = time.time()

#     res = torch.complex(real_part, imag_part)
#     print(f"torch.compile {i} took {et - st}\n")

# opt_foo = torch.compile(foo)

# print(opt_foo(torch.randn(10, 10), torch.randn(10, 10)))

C = torch.matmul(A, B)
CC = torch.tensordot(A, B, dims=1)
print(f"C {C} CC {CC}")

real_part, imag_part = custom_tensordot(A.real, A.imag, B.real, B.imag)
final_tensor = torch.complex(real_part, imag_part).squeeze()
print(f"CCC {final_tensor}")
