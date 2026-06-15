import torch
from scoring.metrics import calculate_enem_metrics
from tqdm import tqdm

def train_one_epoch(model, loader, optmizer, loss_function, device):
    model.train()

    epoch_loss = 0.0
    competences_loss = [0.0, 0.0, 0.0, 0.0, 0.0]

    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        real_grades = batch["grades"][:, :5].float().to(device)

        optmizer.zero_grad()
        preds = model(input_ids, attention_mask)

        batch_loss = 0.0
        for i in range(5):
            loss_c = loss_function(preds[i], real_grades[:, i])
            batch_loss += loss_c
            competences_loss[i] += loss_c.item()

        epoch_loss += batch_loss.item()
        batch_loss.backward()
        optmizer.step()
    
    competences_loss_avg = [c_loss/len(loader) for c_loss in competences_loss]
    return epoch_loss/len(loader), competences_loss_avg

@torch.no_grad()
def evaluate(model, loader, loss_function, device):
    model.eval()

    val_loss = 0.0
    competences_loss = [0.0, 0.0, 0.0, 0.0, 0.0]
    all_preds = {i: [] for i in range(5)}
    all_grades = {i: [] for i in range(5)}

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        real_grades = batch["grades"][:, :5].float().to(device)
    
        preds = model(input_ids, attention_mask)

        batch_loss = 0.0
        for i in range(5):
            loss_c = loss_function(preds[i], real_grades[:, i])
            batch_loss += loss_c
            competences_loss[i] += loss_c.item()
            
            all_preds[i].append(preds[i].cpu())
            all_grades[i].append(real_grades[:, i].cpu())
            
        val_loss += batch_loss.item()
    
    rmses = []
    for i in range(5):
        preds_all_c = torch.cat(all_preds[i])
        grades_all_c = torch.cat(all_grades[i])

        mse = ((preds_all_c - grades_all_c) ** 2).mean()
        rmse_c = torch.sqrt(mse).item() * 40
        rmses.append(rmse_c)

    competences_loss_avg = [c_loss/len(loader) for c_loss in competences_loss]
    return val_loss/len(loader), competences_loss_avg, rmses

def run_training(model, train_loader, val_loader, optimizer, loss_function, epochs, device):
    for epoch in range(1, epochs + 1):
        train_loss, train_comp = train_one_epoch(model, train_loader, optimizer, loss_function, device)
        val_loss, val_comp, val_rmses = evaluate(model, val_loader, loss_function, device)

        print(f"Epoch {epoch}/{epochs}")
        print(f"Global Loss | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        print("-" * 85)
        print(f"Train Loss  | C1: {train_comp[0]:.4f} | C2: {train_comp[1]:.4f} | C3: {train_comp[2]:.4f} | C4: {train_comp[3]:.4f} | C5: {train_comp[4]:.4f}")
        print(f"Val Loss    | C1: {val_comp[0]:.4f} | C2: {val_comp[1]:.4f} | C3: {val_comp[2]:.4f} | C4: {val_comp[3]:.4f} | C5: {val_comp[4]:.4f}")
        print("-" * 85)
        print(f"Val RMSE    | C1: {val_rmses[0]:.1f}   | C2: {val_rmses[1]:.1f}   | C3: {val_rmses[2]:.1f}   | C4: {val_rmses[3]:.1f}   | C5: {val_rmses[4]:.1f}")
        print("=" * 85)

@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()

    all_preds = {i: [] for i in range(5)}
    all_grades = {i: [] for i in range(5)}

    for batch in tqdm(loader, desc="Testing", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        real_grades = batch["grades"][:, :5].float().to(device)
    
        preds = model(input_ids, attention_mask)

        for i in range(5):
            pred_discrete = torch.clamp(preds[i], min=0.0, max=5.0)
            pred_discrete = torch.round(pred_discrete)
            
            all_preds[i].append(pred_discrete.cpu())
            all_grades[i].append(real_grades[:, i].cpu())
    
    final_preds = {i: torch.cat(all_preds[i]).numpy() for i in range(5)}
    final_grades = {i: torch.cat(all_grades[i]).numpy() for i in range(5)}

    return final_grades, final_preds

def run_testing(model, test_loader, device):
    y_true_dict, y_pred_dict = get_predictions(model, test_loader, device)

    final_metrics = {}
    matrices = {}
    width = 85
    print("\n" + "=" * width)
    print(f"{'Evaluation Metrics in the Test Set':^{width}}")
    print("=" * width)
    print(f"{'COMP':^6} | {'QWK':^8} | {'RMSE':^8} | {'F1-MACRO':^10} | {'ACC':^7} | {'ACC_REL':^9} | {'ACC_ENEM':^10}")
    print("-" * width)
    for i in range(5):
        metrics = calculate_enem_metrics(y_true_dict[i], y_pred_dict[i], step=40)

        final_metrics[f"C{i+1}_QWK"] = metrics['QWK']
        final_metrics[f"C{i+1}_RMSE"] = metrics['RMSE']
        final_metrics[f"C{i+1}_F1_Macro"] = metrics['F1_Macro']
        final_metrics[f"C{i+1}_Acc"] = metrics['Acc']
        final_metrics[f"C{i+1}_Acc_Relaxed"] = metrics['Acc_Relaxed']
        final_metrics[f"C{i+1}_Acc_Enem"] = metrics['Acc_Enem']
        final_metrics[f"C{i+1}_Conf_Matrix"] = str(metrics['Confusion_Matrix'].tolist())
        matrices[f"C{i+1}"] = metrics['Confusion_Matrix']

        c_name = f"C{i+1}"
        qwk = f"{metrics['QWK']:.4f}"
        rmse = f"{metrics['RMSE']:.2f}"
        f1 = f"{metrics['F1_Macro']:.4f}"
        acc = f"{metrics['Acc'] * 100:.1f}%"
        acc_rel = f"{metrics['Acc_Relaxed'] * 100:.1f}%"
        acc_enem = f"{metrics['Acc_Enem'] * 100:.1f}%"

        print(f"{c_name:^6} | {qwk:^8} | {rmse:^8} | {f1:^10} | {acc:^7} | {acc_rel:^9} | {acc_enem:^10}")
    print("=" * width + "\n")

    print(f"\n{'Confusion Matrix (Rows: Actual | Columns: Predicted)':^{width}}")
    for c_name, matrix in matrices.items():
        print(f"\n  {c_name}:")
        for row in matrix:
            print("    " + "  ".join([f"{val:4d}" for val in row]))
    print("=" * width + "\n")

    return final_metrics