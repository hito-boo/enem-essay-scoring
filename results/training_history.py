import os
import pandas as pd
import matplotlib.pyplot as plt


class TrainingHistory:
    def __init__(self):
        self.epoch_records = []
        self.step_train_loss = []
        self.step_eval_loss = []
        self.global_step = 0
    
    def add_train_step(self, train_loss):
        self.global_step += 1
        self.step_train_loss.append({"step": self.global_step, "train_loss": train_loss})

    def add_eval_point(self, eval_loss):
        self.step_eval_loss.append({"step": self.global_step, "eval_loss": eval_loss})

    def add_epoch(self, epoch, train_loss, val_loss, train_comp_loss, val_comp_loss, val_rmses, learning_rate):
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rate,
        }
        for i in range(5):
            record[f"train_loss_c{i+1}"] = train_comp_loss[i]
            record[f"val_loss_c{i+1}"] = val_comp_loss[i]
            record[f"val_rmse_c{i+1}"] = val_rmses[i]

        self.epoch_records.append(record)

    def to_dataframe(self):
        return pd.DataFrame(self.epoch_records)

    def train_loss_dataframe(self):
        return pd.DataFrame(self.step_train_loss)

    def eval_loss_dataframe(self):
        return pd.DataFrame(self.step_eval_loss)

    def save_csv(self, filepath="results/training_history/history.csv"):
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        self.to_dataframe().to_csv(filepath, index=False)
        print(f"\n[Training History] Training history saved in: {filepath}")

    def plot(self, filepath="results/training_history/history.png"):
        df = self.to_dataframe()
        if df.empty:
            print("[Training History] No recorded history, graphic not generated...")
            return

        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # Global Loss (Train vs. Validation) per Epoch
        ax = axes[0, 0]
        ax.plot(df["epoch"], df["train_loss"], label="Train", marker="o")
        ax.plot(df["epoch"], df["val_loss"], label="Validation", marker="o")
        ax.set_title("Global Loss per Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)

        # Validation Loss per Competence
        ax = axes[0, 1]
        for i in range(5):
            ax.plot(df["epoch"], df[f"val_loss_c{i+1}"], label=f"C{i+1}", marker="o")
        ax.set_title("Validation Loss per Competence")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)

        # Validation RMSE per Competence
        ax = axes[1, 0]
        for i in range(5):
            ax.plot(df["epoch"], df[f"val_rmse_c{i+1}"], label=f"C{i+1}", marker="o")
        ax.set_title("Validation RMSE per Competence")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE")
        ax.legend()
        ax.grid(alpha=0.3)

        # Learning Rate along Epochs
        ax = axes[1, 1]
        ax.plot(df["epoch"], df["learning_rate"], marker="o", color="tab:red")
        ax.set_title("Learning Rate per Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        print(f"[Training History] Graphic per Epoch saved in: {filepath}")

    def save_learning_curve_csv(self, train_filepath="results/training_history/train_loss_steps.csv", eval_filepath="results/training_history/eval_loss_steps.csv"):
        for filepath, df in ((train_filepath, self.train_loss_dataframe()), (eval_filepath, self.eval_loss_dataframe())):
            dirname = os.path.dirname(filepath)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            df.to_csv(filepath, index=False)

        print(f"[Training History] Learning Rate curve saved in: {train_filepath} and {eval_filepath}")

    def plot_learning_curve(self, filepath="results/training_history/learning_curve.png"):
        train_df = self.train_loss_dataframe()
        eval_df = self.eval_loss_dataframe()

        if train_df.empty:
            print("[Training History] No train step record, Learning Rate curve not generated.")
            return

        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        plt.figure(figsize=(14, 7))
        plt.plot(train_df["step"], train_df["train_loss"], label="Train Loss")
        if not eval_df.empty:
            plt.plot(eval_df["step"], eval_df["eval_loss"], label="Validation Loss", marker="o", linestyle="--")
        plt.title("Learning Curve: Train Loss vs. Validation Loss", fontsize=16)
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"[Training History] Learning Rate curve saved in: {filepath}")