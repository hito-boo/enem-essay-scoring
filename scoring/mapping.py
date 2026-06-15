import torch.nn as nn
import torch

optimizer_mapping = {
    "adamw": torch.optim.AdamW
}

loss_mapping = {
    "mse": nn.MSELoss
}