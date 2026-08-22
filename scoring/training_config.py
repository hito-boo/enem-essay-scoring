from config import CONFIG
from scoring.early_stopping import EarlyStopping

def get_loss_weights():
    weights = CONFIG.get("params", {}).get("loss_weights")

    if weights is None:
        return [1.0] * 5

    if len(weights) != 5:
        raise ValueError(
            f"Error: 'params.loss_weights' must have five weights, but had {len(weights)}."
        )

    if any(w < 0 for w in weights):
        raise ValueError("Error: 'params.loss_weights' must not have negative weights.")

    total = sum(weights)
    if total <= 0:
        raise ValueError("Error: Sum of 'params.loss_weights' must be greater than zero.")

    percentages = [w / total for w in weights]
    scaled_weights = [p * 5 for p in percentages]

    pct_str = ", ".join(f"C{i+1}={p*100:.1f}%" for i, p in enumerate(percentages))
    print(f"\n[Config] Weight of each competence loss: {pct_str}\n")

    return scaled_weights


def get_grad_accum_steps():
    grad_accum_config = CONFIG.get("gradient_accumulation", {})

    if not grad_accum_config.get("enabled", False):
        return 1

    steps = grad_accum_config.get("steps", 1)
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("'gradient_accumulation.steps' must be an integer greater than or equal to 1.")

    return steps


def build_early_stopping():
    es_config = CONFIG.get("early_stopping", {})

    if not es_config.get("enabled", False):
        return None

    return EarlyStopping(
        patience=es_config.get("patience", 3),
        min_delta=es_config.get("min_delta", 0.0),
        checkpoint_dir=es_config.get("checkpoint_dir", "checkpoints"),
        checkpoint_name=es_config.get("checkpoint_name", "best_model.pt"),
        restore_best_weights=es_config.get("restore_best_weights", True),
    )