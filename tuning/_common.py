import copy
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = PROJECT_ROOT / "aesconfig.json"

with BASE_CONFIG_PATH.open("r", encoding="utf-8") as f:
    BASE_CONFIG = json.load(f)

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
    value_str = "-".join(str(v) for v in value) if isinstance(value, list) else str(value)
    return f"{'.'.join(parts)}={value_str}"

def build_trials(baseline, search_space, seeds):
    base_trials = [{"tag": "baseline", "overrides": baseline}]
    for path, values in search_space:
        for value in values:
            base_trials.append({"tag": path_tag(path, value), "overrides": baseline + [(path, value)]})

    trials = []
    for t in base_trials:
        for seed in seeds:
            suffix = "" if len(seeds) == 1 else f"_seed{seed}"
            trials.append({
                "tag": f"{t['tag']}{suffix}",
                "overrides": t["overrides"] + [(("params", "seed"), seed)],
            })
    return trials

def run_trial(trial, index, total, trials_dir, stage_name):
    tag, overrides = trial["tag"], trial["overrides"]
    trial_config = build_trial_config(overrides)
    trial_config["_tuning"] = {"stage": stage_name, "tag": tag}

    trial_config_path = trials_dir / f"{tag}.json"
    with trial_config_path.open("w", encoding="utf-8") as f:
        json.dump(trial_config, f, indent=4, ensure_ascii=False)

    env = os.environ.copy()
    env["AES_CONFIG_PATH"] = str(trial_config_path)

    print(f"\n[{index}/{total}] Running trial: {tag}")
    print(f"  Config: {trial_config_path}")

    subprocess.run([sys.executable, "-m", "scoring.main"], cwd=PROJECT_ROOT, env=env, check=True)

def run_stage(stage_name, baseline, search_space, seeds=(21,)):
    trials_dir = Path(__file__).resolve().parent / "configs" / stage_name
    trials_dir.mkdir(parents=True, exist_ok=True)

    trials = build_trials(baseline, search_space, list(seeds))

    print(f"Stage: {stage_name}")
    print(f"Seeds per trial: {list(seeds)}")
    print(f"Parameters to test: {[path[-1] for path, _ in search_space]}")
    print(f"Number of trials (Baseline + Alternatives): {len(trials)}")
    print(f"Configurations in: {trials_dir}")
    print("Results in: results/experiments_log.csv (line by trial)")

    for i, trial in enumerate(trials, start=1):
        run_trial(trial, i, len(trials), trials_dir, stage_name)
