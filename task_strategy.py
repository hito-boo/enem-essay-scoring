import torch
from coral_pytorch.dataset import corn_label_from_logits
from config import CONFIG

N_CLASSES = 6

VALID_TASK_TYPES = ("regression", "classification", "ordinal")

def get_task_type():
    task_type = CONFIG["task_type"]
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(f"Unknown 'task_type': '{task_type}'. Use one of {VALID_TASK_TYPES}.")
    return task_type


def num_outputs_for_task(task_type):
    if task_type == "regression":
        return 1
    if task_type == "classification":
        return N_CLASSES
    if task_type == "ordinal":
        return N_CLASSES - 1
    raise ValueError(f"Unknown 'task_type': '{task_type}'.")


def regression_targets(true_classes):
    return true_classes.float()

def classification_targets(true_classes):
    return true_classes.long()

def ordinal_targets(true_classes):
    thresholds = torch.arange(N_CLASSES - 1, device=true_classes.device)
    return (true_classes.unsqueeze(1) > thresholds.unsqueeze(0)).float()

TARGET_BUILDERS = {
    "regression": regression_targets,
    "classification": classification_targets,
    "ordinal": ordinal_targets,
}

def prepare_targets(true_classes):
    return TARGET_BUILDERS[get_task_type()](true_classes)

def decode_regression(preds):
    return preds.clamp(min=0.0, max=N_CLASSES - 1).round().long()

def decode_classification(preds):
    return preds.argmax(dim=-1)

def get_ordinal_method():
    return CONFIG.get("ordinal", {}).get("method", "extended_binary")

def decode_ordinal(preds):
    if get_ordinal_method() == "corn":
        return corn_label_from_logits(preds)
    return (torch.sigmoid(preds) > 0.5).sum(dim=-1).long()

DECODERS = {
    "regression": decode_regression,
    "classification": decode_classification,
    "ordinal": decode_ordinal,
}

def decode_predictions(preds):
    return DECODERS[get_task_type()](preds)