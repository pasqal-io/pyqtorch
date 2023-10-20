from __future__ import annotations

import torch
from torch import Tensor


class CustomMatrix(Tensor):
    """Prototype for a custom 2x2 matmul."""

    def __matmul__(self, other: object):
        R11 = torch.matmul(self.real[0, :], other.real[:, 0]) - torch.matmul(
            self.imag[0, :], other.imag[:, 0]
        )
        R12 = torch.matmul(self.real[0, :], other.real[:, 1]) - torch.matmul(
            self.imag[0, :], other.imag[:, 1]
        )
        R21 = torch.matmul(self.real[1, :], other.real[:, 0]) - torch.matmul(
            self.imag[1, :], other.imag[:, 0]
        )
        R22 = torch.matmul(self.real[1, :], other.real[:, 1]) - torch.matmul(
            self.imag[1, :], other.imag[:, 1]
        )
        I11 = torch.matmul(self.real[0, :], other.imag[:, 0]) + torch.matmul(
            self.imag[0, :], other.real[:, 0]
        )
        I12 = torch.matmul(self.real[0, :], other.imag[:, 1]) + torch.matmul(
            self.imag[0, :], other.real[:, 1]
        )
        I21 = torch.matmul(self.real[1, :], other.imag[:, 0]) + torch.matmul(
            self.imag[1, :], other.real[:, 0]
        )
        I22 = torch.matmul(self.real[1, :], other.imag[:, 1]) + torch.matmul(
            self.imag[1, :], other.real[:, 1]
        )
        real_part = torch.tensor([[R11, R12], [R21, R22]])
        imag_part = torch.tensor([[I11, I12], [I21, I22]])
        return torch.complex(real_part, imag_part)


@torch.compile
def custom_matmul(Areal: Tensor, Aimag: Tensor, Breal: Tensor, Bimag: Tensor) -> list:
    R11 = torch.matmul(Areal[0, :], Breal[:, 0]) - torch.matmul(Aimag[0, :], Bimag[:, 0])
    R12 = torch.matmul(Areal[0, :], Breal[:, 1]) - torch.matmul(Aimag[0, :], Bimag[:, 1])
    R21 = torch.matmul(Areal[1, :], Breal[:, 0]) - torch.matmul(Aimag[1, :], Bimag[:, 0])
    R22 = torch.matmul(Areal[1, :], Breal[:, 1]) - torch.matmul(Aimag[1, :], Bimag[:, 1])
    I11 = torch.matmul(Areal[0, :], Bimag[:, 0]) + torch.matmul(Aimag[0, :], Breal[:, 0])
    I12 = torch.matmul(Areal[0, :], Bimag[:, 1]) + torch.matmul(Aimag[0, :], Breal[:, 1])
    I21 = torch.matmul(Areal[1, :], Bimag[:, 0]) + torch.matmul(Aimag[1, :], Breal[:, 0])
    I22 = torch.matmul(Areal[1, :], Bimag[:, 1]) + torch.matmul(Aimag[1, :], Breal[:, 1])
    real_part = torch.tensor([[R11, R12], [R21, R22]])
    imag_part = torch.tensor([[I11, I12], [I21, I22]])
    return real_part, imag_part


@torch.compile
def custom_tensordot(Areal: Tensor, Aimag: Tensor, Breal: Tensor, Bimag: Tensor) -> list:
    R11 = torch.matmul(Areal[0, :], Breal) - torch.matmul(Aimag[0, :], Bimag)
    R21 = torch.matmul(Areal[1, :], Breal) - torch.matmul(Aimag[1, :], Bimag)
    I11 = torch.matmul(Areal[0, :], Bimag) + torch.matmul(Aimag[0, :], Breal)
    I21 = torch.matmul(Areal[1, :], Bimag) + torch.matmul(Aimag[1, :], Breal)
    real_part = torch.tensor([[R11], [R21]])
    imag_part = torch.tensor([[I11], [I21]])
    return real_part, imag_part


# @torch.compile
def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b
