import torch.nn as nn
from mamba_block import MambaBlock

class VIPER(nn.Module):
    def __init__(self, channels=64, seq_len=101):
        super().__init__()
        self.embed = nn.Linear(4, channels)
        self.block1 = MambaBlock(channels)
        self.block2 = MambaBlock(channels)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * seq_len, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)
