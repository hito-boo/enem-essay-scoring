from config import CONFIG
from scoring.mapping import optimizer_mapping

NO_DECAY_KEYWORDS = ("bias", "LayerNorm.weight")


def build_optimizer(model):
    weight_decay = CONFIG["params"]["weight_decay"]

    decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not any(nd in n for nd in NO_DECAY_KEYWORDS)
    ]
    no_decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and any(nd in n for nd in NO_DECAY_KEYWORDS)
    ]

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer_cls = optimizer_mapping[CONFIG["optimizer"]]
    return optimizer_cls(param_groups, lr=CONFIG["params"]["learning_rate"])
