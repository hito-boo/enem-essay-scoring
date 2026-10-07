from tuning._common import run_stage
from tuning.stage1_bertimbau_baseline import baseline_for as bertimbau_baseline_for

ACTIVE_TASK_TYPE = "ordinal"  # definir como a abordagem vencedora
ACTIVE_ROUND = 1

def baseline_for(task_type):
    overrides = [
        o for o in bertimbau_baseline_for(task_type)
        if o[0] not in {("transformers", "bert"), ("tokenizer",)}
    ]
    overrides += [
        (("transformers", "bert"), "PORTULAN/albertina-100m-portuguese-ptbr-encoder"),
        (("tokenizer",), "PORTULAN/albertina-100m-portuguese-ptbr-encoder"),
        (("lora", "target_modules"), ["query_proj", "value_proj"]),
    ]
    return overrides

SEARCH_SPACE_ROUNDS = {
    1: [(("params", "learning_rate"), [1e-5, 2e-5, 3e-5, 5e-5])],
    2: [(("lr_scheduler", "warmup_ratio"), [0.0, 0.2])],
}

if __name__ == "__main__":
    run_stage(
        stage_name=f"stage3_albertina_baseline_{ACTIVE_TASK_TYPE}",
        baseline=baseline_for(ACTIVE_TASK_TYPE),
        search_space=SEARCH_SPACE_ROUNDS[ACTIVE_ROUND],
        seeds=[21],
    )
