import torch
from config import CONFIG
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

# Types: linear_warmup (warmup_ratio), cosine_warmup (warmup_ratio), reduce_on_plateau (factor, patience)

def build_scheduler(optimizer, num_training_steps):
    sched_config = CONFIG.get("lr_scheduler", {})

    if not sched_config.get("enabled", False):
        return None, None

    sched_type = sched_config.get("type", "linear_warmup")

    if sched_type in ("linear_warmup", "cosine_warmup"):
        warmup_ratio = sched_config.get("warmup_ratio", 0.1)
        num_warmup_steps = int(num_training_steps * warmup_ratio)

        if sched_type == "linear_warmup":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        else:
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        return scheduler, "step"

    if sched_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=sched_config.get("factor", 0.5), patience=sched_config.get("patience", 1))
        return scheduler, "epoch"

    raise ValueError(
        f"Unknown 'lr_scheduler.type': '{sched_type}'. Use 'linear_warmup', 'cosine_warmup' or 'reduce_on_plateau'."
    )
