from matrix import CustomMatrix, custom_matmul, foo
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

A = torch.rand(2,2, dtype=torch.complex128)
B = torch.rand(2,2, dtype=torch.complex128)

C = torch.matmul(A,B)
print(C)

AA = CustomMatrix(A)
BB = CustomMatrix(B)

CC = AA @ BB
print(CC)

real_part, imag_part = custom_matmul(A.real, A.imag, B.real, B.imag)
print(torch.complex(real_part, imag_part))

# opt_foo = torch.compile(foo)

# print(opt_foo(torch.randn(10, 10), torch.randn(10, 10)))
