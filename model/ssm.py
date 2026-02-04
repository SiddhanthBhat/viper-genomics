import torch
import torch.nn as nn

class SimpleSSM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.A = nn.Parameter(torch.randn(dim))
        self.B = nn.Linear(dim, dim)
        self.C = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        state = torch.zeros(B, D, device=x.device)
        outputs = []

        for t in range(L):
            state = torch.tanh(self.A * state + self.B(x[:, t]))
            outputs.append(self.C(state))

        return torch.stack(outputs, dim=1)
