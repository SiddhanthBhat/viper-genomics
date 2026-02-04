import torch
import torch.nn as nn
from .mamba_block import ViperBlock

class VIPER(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Hyperparameters
        d_model = config['model']['d_model']
        n_layers = config['model']['n_layers']
        n_classes = config['model']['n_classes']
        vocab_size = 4  # A, C, G, T, (N is masked or 0)

        # 1. Embedding Layer
        # Projects 4-channel one-hot input to d_model dimensions
        self.embedding = nn.Linear(vocab_size, d_model)

        # 2. Backbone: Stack of Viper Blocks
        self.layers = nn.ModuleList([
            ViperBlock(config) for _ in range(n_layers)
        ])

        # 3. Global Pooling
        self.norm_f = nn.LayerNorm(d_model)
        
        # 4. Dense Classification Head
        # "Dense -> ReLU -> Dropout -> Dense -> Sigmoid"
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(config['train']['dropout']),
            nn.Linear(d_model * 2, n_classes),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

    def forward(self, x):
        # x shape: [Batch, Length, 4]
        
        x = self.embedding(x)  # [B, L, d_model]
        
        # Pass through Viper Blocks with residual connections handled inside blocks
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)

        # Global Average Pooling across the sequence length
        # [B, L, D] -> [B, D]
        x = x.mean(dim=1)

        # Binary Classification
        logits = self.classifier(x)
        return logits