import numpy as np
from sklearn.metrics import cohen_kappa_score, root_mean_squared_error, f1_score, confusion_matrix, accuracy_score

def calculate_enem_metrics(y_true, y_pred, step=40):
    y_true_enem = y_true * step
    y_pred_enem = y_pred * step

    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    rmse = root_mean_squared_error(y_true_enem, y_pred_enem)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    
    errors = np.abs(y_true_enem - y_pred_enem)
    acc_relax = np.mean(errors <= 40)
    acc_enem = np.mean(errors <= 80)

    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5])

    return {
        "QWK": qwk,
        "RMSE": rmse,
        "F1_Macro": f1_macro,
        "F1_Weighted": f1_weighted,
        "Acc": acc,
        "Acc_Relaxed": acc_relax,
        "Acc_Enem": acc_enem,
        "Confusion_Matrix": conf_matrix
    }
