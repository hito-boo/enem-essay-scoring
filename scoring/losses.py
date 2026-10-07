import torch
import torch.nn as nn
from config import CONFIG
from scoring.mapping import loss_mapping
from task_strategy import N_CLASSES, get_task_type

class PlainLoss(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, preds, targets, true_classes):
        return self.criterion(preds, targets)

class ClassBalancedLoss(nn.Module):
    def __init__(self, criterion, class_weights):
        super().__init__()
        self.criterion = criterion
        self.register_buffer("class_weights", class_weights)

    def forward(self, preds, targets, true_classes):
        per_sample_loss = self.criterion(preds, targets)
        if per_sample_loss.dim() > 1:
            per_sample_loss = per_sample_loss.mean(dim=-1)
        weights = self.class_weights[true_classes]
        return (per_sample_loss * weights).mean()

def compute_class_weights(train_loader, num_classes=N_CLASSES):
    cb_config = CONFIG.get("class_balanced_loss", {})
    power = cb_config.get("power", 1.0)

    counts = [torch.zeros(num_classes) for _ in range(5)]

    for batch in train_loader:
        grades = batch["grades"][:, :5].round().long().clamp(0, num_classes - 1)
        for i in range(5):
            counts[i] += torch.bincount(grades[:, i], minlength=num_classes).float()

    weights = []
    for i in range(5):
        freq = counts[i].clamp(min=1.0)
        raw_weights = 1.0 / (freq ** power)
        normalized = raw_weights * (num_classes / raw_weights.sum())
        weights.append(normalized)

    return weights

def build_train_loss_functions(train_loader):
    cb_config = CONFIG.get("class_balanced_loss", {})

    if not cb_config.get("enabled", False):
        loss_cls = loss_mapping[CONFIG["loss_function"]]
        return build_eval_loss_functions()

    power = cb_config.get("power", 1.0)
    print(f"[Loss] Class Balanced Loss Activated (Power = {power}) - Weights By Class...")
    class_weights = compute_class_weights(train_loader)
    for i, w in enumerate(class_weights):
        print(f"[Loss] C{i + 1} weight by class (0-5): {[round(x, 2) for x in w.tolist()]}")

    loss_cls = loss_mapping[get_task_type()]
    return [ClassBalancedLoss(loss_cls(reduction="none"), w) for w in class_weights]


def build_eval_loss_functions():
    loss_cls = loss_mapping[get_task_type()]
    return [PlainLoss(loss_cls()) for _ in range(5)]
