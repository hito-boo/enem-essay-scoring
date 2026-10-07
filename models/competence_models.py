import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, dropout=0.0, num_outputs=1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(768, num_outputs)

    def forward(self, cls_repr):
        cls_repr = self.dropout(cls_repr)
        out = self.head(cls_repr)
        return out.squeeze(-1) if out.shape[-1] == 1 else out

class MultiLayerHead(nn.Module):
    def __init__(self, mlp_hidden=256, dropout=0.0, num_outputs=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(768, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_outputs),
        )
    
    def forward(self, cls_repr):
        out = self.head(cls_repr)
        return out.squeeze(-1) if out.shape[-1] == 1 else out
