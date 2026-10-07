import torch.nn as nn
import torch

optimizer_mapping = {
    "adamw": torch.optim.AdamW
}

loss_mapping = {
    "regression": nn.MSELoss,
    "classification": nn.CrossEntropyLoss,
    "ordinal": nn.BCEWithLogitsLoss,
}