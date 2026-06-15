from config import CONFIG, DEVICE, set_seed
from dataset.dataloader import get_dataloaders
from models.model_picker import get_model
from results.results_logger import save_to_csv # , log_to_mlflow
from scoring.engine import run_training, run_testing
from scoring.mapping import optimizer_mapping, loss_mapping

# Set Seed for replicability
set_seed(CONFIG["params"]["seed"])

# Get DataLoaders (check 'dataset' module)
print("Building DataLoaders...")
dataloaders = get_dataloaders()

# Get Model (check 'models' module)
print(f"Building Model and moving it to Device: {DEVICE}...")
model = get_model().to(DEVICE)

# Get Model Optimizer and Loss Function
optimizer = optimizer_mapping[CONFIG["optimizer"]](
    model.parameters(),
    lr=CONFIG["params"]["learning_rate"],
    weight_decay=CONFIG["params"]["weight_decay"],
)
loss_function = loss_mapping[CONFIG["loss_function"]]()

# Train Model (from engine.py)
print("Starting training of the Model...")
run_training(
    model,
    dataloaders["train"],
    dataloaders["validation"],
    optimizer,
    loss_function,
    CONFIG["params"]["epochs"],
    DEVICE,
)

# Test Model (from engine.py)
print("Starting testing of the Model...")
final_results = run_testing(
    model=model, 
    test_loader=dataloaders["test"], 
    device=DEVICE
)

# Save results (check 'results' module)
save_to_csv(CONFIG, final_results)
# log_to_mlflow(CONFIG, final_results)

# Próximos Passos:
# 1. Analisar implementação de Early Stopping com Checkpoint e Avaliação do Treino
# 2. Analisar implementação de Gradient Accumulation
# 3. Analisar implementação do Learning Rate Scheduler
# 4. Analisar implementação de todos esses aspectos para cada cabeça de saída
# X. Verificar agregação do Split de Avaliação no Split de Teste