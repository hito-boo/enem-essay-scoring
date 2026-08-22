import torch
import torch.nn as nn
from config import CONFIG
from scoring.mapping import loss_mapping

N_CLASSES = 6

class ClassBalancedMSELoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(self, preds, targets):
        target_classes = targets.round().long().clamp(0, len(self.class_weights) - 1)
        weights = self.class_weights[target_classes]
        squared_error = (preds - targets) ** 2
        return (squared_error * weights).mean()

def compute_class_weights(train_loader, num_classes=N_CLASSES):
    counts = [torch.zeros(num_classes) for _ in range(5)]

    for batch in train_loader:
        grades = batch["grades"][:, :5].round().long().clamp(0, num_classes - 1)
        for i in range(5):
            counts[i] += torch.bincount(grades[:, i], minlength=num_classes).float()

    weights = []
    for i in range(5):
        freq = counts[i].clamp(min=1.0)
        inv_freq = 1.0 / freq
        normalized = inv_freq * (num_classes / inv_freq.sum())
        weights.append(normalized)

    return weights

def build_train_loss_functions(train_loader):
    cb_config = CONFIG.get("class_balanced_loss", {})

    if not cb_config.get("enabled", False):
        loss_cls = loss_mapping[CONFIG["loss_function"]]
        return [loss_cls() for _ in range(5)]

    print("[Loss] Class Balanced Loss Activated - Weights By Class (Inverse Frequeny)...")
    class_weights = compute_class_weights(train_loader)
    for i, w in enumerate(class_weights):
        print(f"[Loss] C{i + 1} weight by class (0-5): {[round(x, 2) for x in w.tolist()]}")

    return [ClassBalancedMSELoss(w) for w in class_weights]


def build_eval_loss_functions():
    loss_cls = loss_mapping[CONFIG["loss_function"]]
    return [loss_cls() for _ in range(5)]
