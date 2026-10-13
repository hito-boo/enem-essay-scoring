import pandas as pd

stage = "stage2_bertimbau_lora"

df = pd.read_csv("results/experiments_log.csv")
rodada = df[df["tuning_stage"] == stage]
print(rodada.sort_values("val_qwk_mean", ascending=False)[["tuning_tag", "val_qwk_mean", "val_loss"]])