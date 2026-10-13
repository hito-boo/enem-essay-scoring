from tuning._common import run_stage
from tuning.stage1_bertimbau_baseline import baseline_for

ACTIVE_ROUND = 0

BERTIMBAU_BASELINE = baseline_for("ordinal", "extended_binary") + [

    (("lora", "enabled"), True),
    (("lora", "r"), 8),
    (("lora", "alpha"), 16),
    (("lora", "dropout"), 0.1),
    (("lora", "target_modules"), ["query", "value"]),
    (("gradient_clipping", "enabled"), True),
    (("gradient_clipping", "max_norm"), 1.0),
    (("params", "learning_rate"), 4e-4),
]

SEARCH_SPACE_ROUNDS = {
    0: [],
    1: [(("params", "learning_rate"), [2e-4, 4e-4, 6e-4, 8e-4, 1e-3, 1.5e-3])],
    2: [(("lora", "r"), [16])],
}

if __name__ == "__main__":
    run_stage(
        stage_name="stage2_bertimbau_lora",
        baseline=BERTIMBAU_BASELINE,
        search_space=SEARCH_SPACE_ROUNDS[ACTIVE_ROUND],
        seeds=[21],
    )