import os
import pandas as pd
import mlflow
from datetime import datetime

def flatten_config(config):
    flat_config = {}
    
    for key, value in config.get("transformers", {}).items():
        flat_config[f"arch_{key}"] = value
        
    flat_config["optimizer"] = config.get("optimizer", "N/A")
    flat_config["loss_function"] = config.get("loss_function", "N/A")

    for key, value in config.get("params", {}).items():
        if isinstance(value, list) and len(value) == 6:
            flat_config[f"{key}_body"] = value[0]
            for i in range(1, 6):
                flat_config[f"{key}_c{i}"] = value[i]
        elif isinstance(value, list) and len(value) == 5:
            for i in range(5):
                flat_config[f"{key}_c{i+1}"] = value[i]
        else:
            flat_config[key] = value

    for section in ("early_stopping", "gradient_accumulation", "lr_scheduler"):
        for key, value in config.get(section, {}).items():
            flat_config[f"{section}_{key}"] = value

    return flat_config

def save_to_csv(config, metrics_dict, filepath="results/experiments_log.csv"):
    row_data = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    row_data.update(flatten_config(config))
    row_data.update(metrics_dict)
    
    df_new = pd.DataFrame([row_data])
    
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


    if os.path.exists(filepath):
        df_existing = pd.read_csv(filepath)
        df_final = pd.concat([df_existing, df_new], ignore_index=True) 
        df_final.to_csv(filepath, index=False)
    else:
        df_new.to_csv(filepath, index=False)
        
    print(f"\n[Results Logger] Results successfully saved in: {filepath}")

def log_to_mlflow(config, metrics_dict, experiment_name="Enem-Essay-Scoring"):
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        mlflow.log_params(flatten_config(config))
        
        numerical_metrics = {k: v for k, v in metrics_dict.items() if not isinstance(v, str)}
        string_metrics = {k: v for k, v in metrics_dict.items() if isinstance(v, str)}
        
        mlflow.log_metrics(numerical_metrics)
        
        for key, text_value in string_metrics.items():
            mlflow.log_text(text_value, f"confusion_matrices/{key}.txt")
            
    print(f"[Logger] Experiment recorded in MLflow (Experiment: {experiment_name})")