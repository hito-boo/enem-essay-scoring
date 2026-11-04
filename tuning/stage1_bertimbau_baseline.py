from tuning._common import run_stage

ACTIVE_TASK_TYPE = "ordinal"
ACTIVE_ORDINAL_METHOD = "extended_binary" # or "corn"
ACTIVE_ROUND = 0

def baseline_for(task_type, ordinal_method):
    return [
        (("transformers", "body"), "base"),
        (("transformers", "bert"), "neuralmind/bert-base-portuguese-cased"),
        (("tokenizer",), "neuralmind/bert-base-portuguese-cased"),
        (("lora", "enabled"), False),
        (("task_type",), task_type),
        (("ordinal", "method"), ordinal_method),
        (("params", "learning_rate"), 5e-5), # 3e-5
        (("params", "weight_decay"), 0.01),
        (("params", "batch_size"), 8),
        (("params", "epochs"), 10),
        (("params", "drop_out"), [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (("params", "loss_weights"), [0.18, 0.21, 0.21, 0.18, 0.22]),
        (("class_balanced_loss", "enabled"), True),
        (("class_balanced_loss", "power"), 1.0),
        (("early_stopping", "enabled"), True),
        (("early_stopping", "patience"), 3),
        (("early_stopping", "min_delta"), 0.0001),
        (("gradient_accumulation", "enabled"), True),
        (("gradient_accumulation", "steps"), 4),
        (("lr_scheduler", "enabled"), False),
        (("lr_scheduler", "type"), "linear_warmup"),
        (("lr_scheduler", "warmup_ratio"), 0.1),
    ]

def baseline_with_multi_heads(task_type, ordinal_method):
    overrides = [
        o for o in baseline_for(task_type, ordinal_method)
        if o[0] not in {("transformers", "c1"), ("transformers", "c2"), ("transformers", "c3"),
                         ("transformers", "c4"), ("transformers", "c5")}
    ]
    overrides += [
        (("transformers", "c1"), "multi"),
        (("transformers", "c2"), "multi"),
        (("transformers", "c3"), "multi"),
        (("transformers", "c4"), "multi"),
        (("transformers", "c5"), "multi"),
        (("params", "drop_out"), [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    ]
    return overrides

SEARCH_SPACE_ROUNDS = {
    0: [],
    1: [(("params", "learning_rate"), [1e-5, 2e-5, 3e-5])],
    2: [
        (("params", "epochs"), [8, 12]),
        (("lr_scheduler", "enabled"), [True]), # False
        (("gradient_accumulation", "enabled"), [False]),
    ],
    3: [
        (("params", "epochs"), [8, 12]),
        (("gradient_accumulation", "steps"), [4]),
    ],
    4: [
        (("params", "weight_decay"), [0.0, 0.1]),
        (("params", "drop_out", 0), [0.0, 0.2]),
    ],
    5: [(("class_balanced_loss", "power"), [0.0, 0.25, 0.75, 1.0])],
    6: [
        (("class_balanced_loss", "enabled"), [False]),
        (("params", "loss_weights"), [[0.2, 0.2, 0.2, 0.2, 0.2]]),
    ],
    7: [
        (("params", "loss_weights"), [
            [0.15, 0.30, 0.25, 0.15, 0.15],
            [0.15, 0.35, 0.20, 0.15, 0.15],
            [0.15, 0.25, 0.25, 0.15, 0.20],
        ]),
    ],
}

_stage_suffix = ACTIVE_TASK_TYPE
if ACTIVE_TASK_TYPE == "ordinal" and ACTIVE_ORDINAL_METHOD == "corn":
    _stage_suffix = f"{ACTIVE_TASK_TYPE}_corn"

if __name__ == "__main__":
    run_stage(
        stage_name=f"stage1_bertimbau_baseline_{_stage_suffix}",
        baseline=baseline_for(ACTIVE_TASK_TYPE, ACTIVE_ORDINAL_METHOD),
        search_space=SEARCH_SPACE_ROUNDS[ACTIVE_ROUND],
        seeds=[21, 42, 7],
    )
