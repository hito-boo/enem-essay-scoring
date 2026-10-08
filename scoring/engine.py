import torch
from config import CONFIG
from task_strategy import prepare_targets, decode_predictions
from profiling.profiler import ResourceProfiler
from scoring.metrics import calculate_enem_metrics
from scoring.training_config import get_loss_weights, get_grad_accum_steps, build_early_stopping
from scoring.losses import build_train_loss_functions, build_eval_loss_functions
from results.training_history import TrainingHistory
from tqdm import tqdm

def train_one_epoch(model, loader, optmizer, loss_functions, device, loss_weights, grad_accum_steps=1, scheduler=None, scheduler_step_mode=None, history=None):
    model.train()

    epoch_loss = 0.0
    competences_loss = [0.0, 0.0, 0.0, 0.0, 0.0]

    num_batches = len(loader)
    optmizer.zero_grad()

    window_loss_sum = 0.0
    window_count = 0

    for step, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        true_grades = batch["grades"][:, :5].long().to(device)

        preds = model(input_ids, attention_mask)

        batch_loss = 0.0
        for i in range(5):
            targets_i = prepare_targets(true_grades[:, i])
            loss_c = loss_functions[i](preds[i], targets_i, true_grades[:, i])
            batch_loss += loss_weights[i] * loss_c
            competences_loss[i] += loss_c.item()

        epoch_loss += batch_loss.item()
        window_loss_sum += batch_loss.item()
        window_count += 1

        (batch_loss / grad_accum_steps).backward()

        is_last_batch = (step + 1) == num_batches
        if (step + 1) % grad_accum_steps == 0 or is_last_batch:
            optmizer.step()
            if scheduler is not None and scheduler_step_mode == "step":
                scheduler.step()
            optmizer.zero_grad()

            if history is not None:
                history.add_train_step(window_loss_sum / window_count)
            window_loss_sum = 0.0
            window_count = 0
    
    competences_loss_avg = [c_loss/len(loader) for c_loss in competences_loss]
    return epoch_loss/len(loader), competences_loss_avg

@torch.no_grad()
def evaluate(model, loader, loss_functions, device, loss_weights):
    model.eval()

    val_loss = 0.0
    competences_loss = [0.0, 0.0, 0.0, 0.0, 0.0]
    all_preds = {i: [] for i in range(5)}
    all_grades = {i: [] for i in range(5)}

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        true_grades = batch["grades"][:, :5].long().to(device)
    
        preds = model(input_ids, attention_mask)

        batch_loss = 0.0
        for i in range(5):
            targets_i = prepare_targets(true_grades[:, i])
            loss_c = loss_functions[i](preds[i], targets_i, true_grades[:, i])
            batch_loss += loss_weights[i] * loss_c
            competences_loss[i] += loss_c.item()
            
            all_preds[i].append(decode_predictions(preds[i]).cpu())
            all_grades[i].append(true_grades[:, i].cpu())
            
        val_loss += batch_loss.item()
    
    rmses = []
    qwks = []
    for i in range(5):
        preds_all_c = torch.cat(all_preds[i]).float()
        grades_all_c = torch.cat(all_grades[i]).float()

        mse = ((preds_all_c - grades_all_c) ** 2).mean()
        rmse_c = torch.sqrt(mse).item() * 40
        rmses.append(rmse_c)

        qwk_c = calculate_enem_metrics(grades_all_c.numpy().astype(int), preds_all_c.numpy().astype(int), step=40)["QWK"]
        qwks.append(qwk_c)

    competences_loss_avg = [c_loss/len(loader) for c_loss in competences_loss]
    return val_loss/len(loader), competences_loss_avg, rmses, qwks

def run_training(model, train_loader, val_loader, optimizer, epochs, device, scheduler=None, scheduler_step_mode=None):
    loss_weights = get_loss_weights()
    grad_accum_steps = get_grad_accum_steps()
    early_stopping = build_early_stopping()
    history = TrainingHistory()
    hardware_metrics = {}
    best_val_metrics = {}

    train_loss_functions = [fn.to(device) for fn in build_train_loss_functions(train_loader)]
    eval_loss_functions = [fn.to(device) for fn in build_eval_loss_functions()]

    if CONFIG["profiling"]:
        print("\n[Profiler] Starting Profiler")
        profiler = ResourceProfiler(interval=1.0)
        profiler.start()

    for epoch in range(1, epochs + 1):
        train_loss, train_comp = train_one_epoch(
            model, train_loader, optimizer, train_loss_functions, device,
            loss_weights=loss_weights,
            grad_accum_steps=grad_accum_steps,
            scheduler=scheduler,
            scheduler_step_mode=scheduler_step_mode,
            history=history
        )
        val_loss, val_comp, val_rmses, val_qwks = evaluate(model, val_loader, eval_loss_functions, device, loss_weights=loss_weights)

        current_lr = optimizer.param_groups[0]["lr"]
        history.add_epoch(epoch, train_loss, val_loss, train_comp, val_comp, val_rmses, val_qwks, current_lr)
        history.add_eval_point(val_loss)

        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Global Loss | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        print("-" * 85)
        print(f"Train Loss  | C1: {train_comp[0]:.4f} | C2: {train_comp[1]:.4f} | C3: {train_comp[2]:.4f} | C4: {train_comp[3]:.4f} | C5: {train_comp[4]:.4f}")
        print(f"Val Loss    | C1: {val_comp[0]:.4f}   | C2: {val_comp[1]:.4f}   | C3: {val_comp[2]:.4f}   | C4: {val_comp[3]:.4f}   | C5: {val_comp[4]:.4f}")
        print("-" * 85)
        print(f"Val QWK     | C1: {val_qwks[0]:.2f}   | C2: {val_qwks[1]:.2f}   | C3: {val_qwks[2]:.2f}   | C4: {val_qwks[3]:.2f}   | C5: {val_qwks[4]:.1f}")
        print(f"Val RMSE    | C1: {val_rmses[0]:.1f}   | C2: {val_rmses[1]:.1f}   | C3: {val_rmses[2]:.1f}   | C4: {val_rmses[3]:.1f}   | C5: {val_rmses[4]:.1f}")
        print(f"Learning Rate: {current_lr:.2e}")
        print("=" * 85)

        if scheduler is not None and scheduler_step_mode == "epoch":
            scheduler.step(val_loss)

        def snapshot_best_val(val_loss, val_rmses, val_qwks):
            snapshot = {"val_loss": val_loss, "val_qwk_mean": sum(val_qwks) / len(val_qwks)}
            for i in range(5):
                snapshot[f"val_qwk_c{i+1}"] = val_qwks[i]
                snapshot[f"val_rmse_c{i+1}"] = val_rmses[i]
            return snapshot

        if early_stopping is not None:
            improved = early_stopping.step(val_loss, model, epoch)
            if improved:
                best_val_metrics = snapshot_best_val(val_loss, val_rmses, val_qwks)
                print(f"[Early Stopping] New best model saved in '{early_stopping.checkpoint_path}' (epoch {epoch}, val_loss={val_loss:.4f}).")
            else:
                print(f"[Early Stopping] No improvement in {early_stopping.counter}/{early_stopping.patience} epoch(s)")

            if early_stopping.should_stop:
                print(f"[Early Stopping] Training interrupted in epoch {epoch}. Best epoch: {early_stopping.best_epoch} (val_loss={early_stopping.best_score:.4f})")
                print("=" * 85)
                break
        else:
            best_val_metrics = snapshot_best_val(val_loss, val_rmses, val_qwks)
    
    if early_stopping is not None and early_stopping.restore_best_weights:
        model = early_stopping.load_best_weights(model, map_location=device)
        print(f"[Early Stopping] Weights from the best checkpoint (epoch {early_stopping.best_epoch}) restored to the model.")

        if CONFIG["profiling"]:
            profiler.stop()
            hardware_metrics = profiler.get_results()
            print(f"\n[Profiler] Time in seconds: {hardware_metrics['time_seconds']:.2f}s")
            print(f"[Profiler] CPU Average Usage: {hardware_metrics['avg_cpu_percent']:.1f}% | Peak: {hardware_metrics['peak_cpu_percent']:.1f}%")
            print(f"[Profiler] RAM Average: {hardware_metrics['avg_ram_mb']:.1f} MB | Peak: {hardware_metrics['peak_ram_mb']:.1f} MB")
            print(f"[Profiler] Peak VRAM: {hardware_metrics['peak_vram_mb']:.1f} MB")
        
    return history, hardware_metrics, best_val_metrics

@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()

    all_preds = {i: [] for i in range(5)}
    all_grades = {i: [] for i in range(5)}

    for batch in tqdm(loader, desc="Testing", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        true_grades = batch["grades"][:, :5].long().to(device)
    
        preds = model(input_ids, attention_mask)

        for i in range(5):
            pred_discrete = decode_predictions(preds[i])
            
            all_preds[i].append(pred_discrete.cpu())
            all_grades[i].append(true_grades[:, i].cpu())
    
    final_preds = {i: torch.cat(all_preds[i]).numpy() for i in range(5)}
    final_grades = {i: torch.cat(all_grades[i]).numpy() for i in range(5)}

    return final_grades, final_preds

def run_testing(model, test_loader, device):
    y_true_dict, y_pred_dict = get_predictions(model, test_loader, device)

    final_metrics = {}
    matrices = {}
    width = 98
    print("\n" + "=" * width)
    print(f"{'Evaluation Metrics in the Test Set':^{width}}")
    print("=" * width)
    print(f"{'COMP':^6} | {'QWK':^8} | {'RMSE':^8} | {'F1-MACRO':^10} | {'F1-WEIGHT':^10} | {'ACC':^7} | {'ACC_REL':^9} | {'ACC_ENEM':^10}")
    print("-" * width)
    for i in range(5):
        metrics = calculate_enem_metrics(y_true_dict[i], y_pred_dict[i], step=40)

        final_metrics[f"C{i+1}_QWK"] = metrics['QWK']
        final_metrics[f"C{i+1}_RMSE"] = metrics['RMSE']
        final_metrics[f"C{i+1}_F1_Macro"] = metrics['F1_Macro']
        final_metrics[f"C{i+1}_F1_Weighted"] = metrics['F1_Weighted']
        final_metrics[f"C{i+1}_Acc"] = metrics['Acc']
        final_metrics[f"C{i+1}_Acc_Relaxed"] = metrics['Acc_Relaxed']
        final_metrics[f"C{i+1}_Acc_Enem"] = metrics['Acc_Enem']
        final_metrics[f"C{i+1}_Conf_Matrix"] = str(metrics['Confusion_Matrix'].tolist())
        matrices[f"C{i+1}"] = metrics['Confusion_Matrix']

        c_name = f"C{i+1}"
        qwk = f"{metrics['QWK']:.4f}"
        rmse = f"{metrics['RMSE']:.2f}"
        f1_macro = f"{metrics['F1_Macro']:.4f}"
        f1_weighted = f"{metrics['F1_Weighted']:.4f}"
        acc = f"{metrics['Acc'] * 100:.1f}%"
        acc_rel = f"{metrics['Acc_Relaxed'] * 100:.1f}%"
        acc_enem = f"{metrics['Acc_Enem'] * 100:.1f}%"

        print(f"{c_name:^6} | {qwk:^8} | {rmse:^8} | {f1_macro:^10} | {f1_weighted:^10} | {acc:^7} | {acc_rel:^9} | {acc_enem:^10}")
    print("=" * width + "\n")

    print(f"\n{'Confusion Matrix (Rows: Actual | Columns: Predicted)':^{width}}")
    for c_name, matrix in matrices.items():
        print(f"\n  {c_name}:")
        for row in matrix:
            print("    " + "  ".join([f"{val:4d}" for val in row]))
    print("=" * width + "\n")

    return final_metrics