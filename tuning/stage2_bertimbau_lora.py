from tuning._common import run_stage
from tuning.stage1_bertimbau_baseline import baseline_for

ACTIVE_ROUND = 1

# Trocar "ordinal" pela abordagem melhor sucedida
BERTIMBAU_BASELINE = baseline_for("ordinal") + [
    (("lora", "enabled"), True),
    (("lora", "r"), 8),
    (("lora", "alpha"), 16),
    (("lora", "dropout"), 0.1),
    (("lora", "target_modules"), ["query", "value"]),
]

SEARCH_SPACE_ROUNDS = {
    1: [(("params", "learning_rate"), [1e-4, 2e-4, 5e-4, 1e-3])],
}

if __name__ == "__main__":
    run_stage(
        stage_name="stage2_bertimbau_lora",
        baseline=BERTIMBAU_BASELINE,
        search_space=SEARCH_SPACE_ROUNDS[ACTIVE_ROUND],
        seeds=[21],
    )