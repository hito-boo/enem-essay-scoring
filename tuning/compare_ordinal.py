import pandas as pd
df = pd.read_csv("results/experiments_log.csv")

old = df[(df["tuning_stage"] == "stage1_bertimbau_baseline_ordinal") & (df["tuning_tag"] == "baseline")]
new = df[(df["tuning_stage"] == "stage1_bertimbau_baseline_ordinal_corn") & (df["tuning_tag"] == "baseline")]

print("Old:")
print(old[["val_qwk_mean", "val_loss"]].to_string(index=False))
print("\nNew:")
print(new[["val_qwk_mean", "val_loss"]].to_string(index=False))