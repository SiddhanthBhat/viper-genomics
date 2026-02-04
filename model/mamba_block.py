import torch
import torch.nn as nn

# Try importing the official efficient Mamba implementation
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("Warning: 'mamba_ssm' not found. VIPER will use a slower, fallback SSM.")

class ViperBlock(nn.Module):
    """
    ViperBlock: A hybrid block combining Local Conv1D pattern extraction
    with Global Mamba State Space modeling.
    Structure: Norm -> Conv1D -> Mamba -> Residual
    """
    def __init__(self, config):
        super().__init__()
        d_model = config['model']['d_model']
        d_state = config['model']['d_state']
        d_conv = config['model']['kernel_size'] # 7 according to paper
        expand = config['model']['expand']

        # Group Normalization (Paper: "Group Normalization + ReLU")
        # Note: Mamba usually puts Norm before the block (Pre-Norm).
        self.norm = nn.LayerNorm(d_model)

        if HAS_MAMBA:
            # Official implementation handles the Conv1d -> SSM internal path efficiently
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,    # Local Conv1D kernel size (7)
                expand=expand,
            )
        else:
            # Fallback for CPU/No-CUDA testing (simplified)
            from .ssm import SimpleSSM
            self.mamba = SimpleSSM(d_model, d_state, d_conv)

    def forward(self, x):
        # Residual connection
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return residual + x