import torch.nn as nn
from ssm import SimpleSSM

class MambaBlock(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2)
        self.norm = nn.GroupNorm(4, dim)
        self.ssm = SimpleSSM(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        z = self.conv(x.transpose(1,2)).transpose(1,2)
        z = self.norm(z.transpose(1,2)).transpose(1,2)
        z = self.act(z)
        s = self.ssm(z)
        return z * s
