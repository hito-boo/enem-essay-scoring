import os
import torch


class EarlyStopping:
    def __init__(
        self,
        patience=3,
        min_delta=0.0,
        checkpoint_dir="checkpoints",
        checkpoint_name="best_model.pt",
        restore_best_weights=True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_score = float("inf")
        self.best_epoch = None
        self.counter = 0
        self.should_stop = False

    def step(self, current_score, model, epoch):
        improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            self._save_checkpoint(model, epoch, current_score)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return improved

    def _save_checkpoint(self, model, epoch, score):
        torch.save(
            {
                "epoch": epoch,
                "val_loss": score,
                "model_state_dict": model.state_dict(),
            },
            self.checkpoint_path,
        )

    def load_best_weights(self, model, map_location=None):
        if not os.path.exists(self.checkpoint_path):
            return model

        checkpoint = torch.load(self.checkpoint_path, map_location=map_location, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
