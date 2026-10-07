from tuning._common import run_stage
from tuning.stage3_albertina_baseline import baseline_for as albertina_baseline_for

# Alterar a abordagem para a vencedora

ALBERTINA_BASELINE = albertina_baseline_for("ordinal") + [
    (("lora", "enabled"), True),
    (("lora", "r"), 8),
    (("lora", "alpha"), 16),
    (("lora", "dropout"), 0.1),
]

SEARCH_SPACE_ROUNDS = {
    1: [(("params", "learning_rate"), [1e-4, 2e-4, 5e-4, 1e-3])],
}

ACTIVE_ROUND = 1

if __name__ == "__main__":
    run_stage(
        stage_name="stage4_albertina_lora",
        baseline=ALBERTINA_BASELINE,
        search_space=SEARCH_SPACE_ROUNDS[ACTIVE_ROUND],
        seeds=[21],
    )