from config import CONFIG, DEVICE, set_seed
from dataset.dataloader import get_dataloaders
from models.model_picker import get_model
from results.results_logger import save_to_csv, log_to_mlflow
from scoring.engine import run_training, run_testing
from scoring.optimizer import build_optimizer
from scoring.scheduler import build_scheduler
from scoring.training_config import get_grad_accum_steps

# Set Seed for replicability
set_seed(CONFIG["params"]["seed"])

# Get DataLoaders (check 'dataset' module)
print("\n[Main] Building DataLoaders...")
dataloaders = get_dataloaders()

# Get Model (check 'models' module)
print(f"[Main] Building Model and moving it to Device: {DEVICE}...")
model = get_model().to(DEVICE)

# Get Model Optimizer (check 'scoring/optimizer.py')
optimizer = build_optimizer(model)

# Get Learning Rate Scheduler, if enabled (check 'scoring/scheduler.py')
grad_accum_steps = get_grad_accum_steps()
steps_per_epoch = -(-len(dataloaders["train"]) // grad_accum_steps) # Ceil division
num_training_steps = steps_per_epoch * CONFIG["params"]["epochs"]
scheduler, scheduler_step_mode = build_scheduler(optimizer, num_training_steps)

# Train Model (from engine.py; loss functions built internally, check 'scoring/losses.py')
print("[Main] Starting training of the Model...")
history, hardware_metrics = run_training(
    model,
    dataloaders["train"],
    dataloaders["validation"],
    optimizer,
    CONFIG["params"]["epochs"],
    DEVICE,
    scheduler=scheduler,
    scheduler_step_mode=scheduler_step_mode,
)

# Save Training History (check 'results/training_history.py')
history_dir = CONFIG.get("training_evaluation", {}).get("output_dir", "results/training_history")
history.save_csv(f"{history_dir}/history.csv")
history.plot(f"{history_dir}/history.png")
history.save_learning_curve_csv(f"{history_dir}/train_loss_steps.csv", f"{history_dir}/eval_loss_steps.csv")
history.plot_learning_curve(f"{history_dir}/learning_curve.png")

# Test Model (from engine.py)
print("\n[Main] Starting testing of the Model...")
final_results = run_testing(
    model=model, 
    test_loader=dataloaders["test"], 
    device=DEVICE
)

# Save results (check 'results' module)
save_to_csv(CONFIG, final_results, hardware_metrics)
log_to_mlflow(CONFIG, final_results, hardware_metrics)

# python -m scoring.main
