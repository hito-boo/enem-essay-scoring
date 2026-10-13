from tuning._common import run_stage
from tuning.stage1_bertimbau_baseline import baseline_for as bertimbau_baseline_for

ACTIVE_TASK_TYPE = "ordinal"
ACTIVE_ORDINAL_METHOD = "extended_binary"
ACTIVE_ROUND = 1

def baseline_for(task_type, ordinal_method):
    overrides = [
        o for o in bertimbau_baseline_for(task_type, ordinal_method)
        if o[0] not in {("transformers", "bert"), ("tokenizer",)}
    ]
    overrides += [
        (("transformers", "bert"), "PORTULAN/albertina-100m-portuguese-ptbr-encoder"),
        (("tokenizer",), "PORTULAN/albertina-100m-portuguese-ptbr-encoder"),
        (("lora", "enabled"), True),
        (("lora", "r"), 8),
        (("lora", "alpha"), 16),
        (("lora", "dropout"), 0.1),
        (("lora", "target_modules"), ["in_proj"]),
    ]
    return overrides

SEARCH_SPACE_ROUNDS = {
    0: [],
    1: [(("params", "learning_rate"), [1e-5, 2e-5, 3e-5])],
}

if __name__ == "__main__":
    run_stage(
        stage_name=f"stage3_albertina_baseline_{ACTIVE_TASK_TYPE}_{ACTIVE_ORDINAL_METHOD}",
        baseline=baseline_for(ACTIVE_TASK_TYPE, ACTIVE_ORDINAL_METHOD),
        search_space=SEARCH_SPACE_ROUNDS[ACTIVE_ROUND],
        seeds=[21],
    )
