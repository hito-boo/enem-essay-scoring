import copy
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = PROJECT_ROOT / "aesconfig.json"
TRIALS_DIR = Path(__file__).resolve().parent / "configs"
TRIALS_DIR.mkdir(parents=True, exist_ok=True)

with BASE_CONFIG_PATH.open("r", encoding="utf-8") as f:
    BASE_CONFIG = json.load(f)

BASELINE = [
    (("params", "learning_rate"), 2e-5),
    (("params", "weight_decay"), 0.01),
    (("params", "batch_size"), 8),
    (("params", "epochs"), 10),
    (("params", "drop_out"), [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (("params", "loss_weights"), [0.20, 0.20, 0.20, 0.20, 0.20]), # [0.18, 0.21, 0.21, 0.18, 0.22]
    (("class_balanced_loss", "enabled"), True),
    (("class_balanced_loss", "power"), 0.7),
    (("early_stopping", "enabled"), True),
    (("early_stopping", "patience"), 3),
    (("early_stopping", "min_delta"), 0.0001),
    (("gradient_accumulation", "enabled"), True),
    (("gradient_accumulation", "steps"), 2),
    (("lr_scheduler", "enabled"), True),
    (("lr_scheduler", "type"), "linear_warmup"),
    (("lr_scheduler", "warmup_ratio"), 0.1),
]

SEARCH_SPACE_ROUNDS = {
    1: [
        (("params", "learning_rate"), [1e-5, 3e-5, 5e-5]),
    ],

    2: [
        (("params", "epochs"), [8, 12]),
        (("lr_scheduler", "enabled"), [False]),
        (("gradient_accumulation", "enabled"), [False]),
    ],

    3: [
        (("lr_scheduler", "type"), ["cosine_warmup"]),
        (("lr_scheduler", "warmup_ratio"), [0.0, 0.2]),
        (("gradient_accumulation", "steps"), [4]),
    ],

    4: [
        (("params", "weight_decay"), [0.0, 0.1]),
        (("params", "drop_out", 0), [0.0, 0.2]),
    ],

    5: [
        (("class_balanced_loss", "power"), [0.0, 0.25, 0.75, 1.0]),
    ],

    6: [
        (("class_balanced_loss", "enabled"), [False]),
        (("params", "loss_weights"), [[0.2, 0.2, 0.2, 0.2, 0.2]]),
    ],
}

ACTIVE_ROUND = 5

SEARCH_SPACE = SEARCH_SPACE_ROUNDS[ACTIVE_ROUND]

def deep_set(config, path, value):
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = copy.deepcopy(value)

def build_trial_config(overrides):
    cfg = copy.deepcopy(BASE_CONFIG)
    for path, value in overrides:
        deep_set(cfg, path, value)
    return cfg

def path_tag(path, value):
    parts = []
    for key in path:
        if isinstance(key, int):
            parts[-1] = f"{parts[-1]}[{key}]"
        else:
            parts.append(str(key))
    return f"{'.'.join(parts)}={value}"

def build_trials():
    trials = [{"tag": "baseline", "overrides": BASELINE}]

    for path, values in SEARCH_SPACE:
        for value in values:
            tag = path_tag(path, value)
            overrides = BASELINE + [(path, value)]
            trials.append({"tag": tag, "overrides": overrides})

    return trials

def run_trial(trial, index, total):
    tag, overrides = trial["tag"], trial["overrides"]
    trial_config = build_trial_config(overrides)

    trial_config_path = TRIALS_DIR / f"{tag}.json"
    with trial_config_path.open("w", encoding="utf-8") as f:
        json.dump(trial_config, f, indent=4, ensure_ascii=False)

    env = os.environ.copy()
    env["AES_CONFIG_PATH"] = str(trial_config_path)

    print()
    print(f"[{index}/{total}] Running trial: {tag}")
    print(f"  Config: {trial_config_path}")

    subprocess.run(
        [sys.executable, "-m", "scoring.main"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )

def main():
    trials = build_trials()

    print(f"Active round: {ACTIVE_ROUND}")
    print(f"Parameters to test: {[path[-1] for path, _ in SEARCH_SPACE]}")
    print(f"Number of trials (Baseline + Alternatives): {len(trials)}")
    print(f"Configurations in: {TRIALS_DIR}")
    print("Results in: results/experiments_log.csv (line by trial)")

    for i, trial in enumerate(trials, start=1):
        run_trial(trial, i, len(trials))

if __name__ == "__main__":
    main()
