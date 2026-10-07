import os
import tempfile
import torch
from config import CONFIG, set_seed
from dataset.dataloader import get_dataloaders
from models.model_picker import get_model
from scoring.engine import run_testing
from profiling.profiler import ResourceProfiler
from results.results_logger import save_to_csv, log_to_mlflow

DEVICE = torch.device("cpu")

def model_size_mb(model):
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        torch.save(model.state_dict(), tmp.name)
        return os.path.getsize(tmp.name) / (1024 ** 2)

def main():
    set_seed(CONFIG["params"]["seed"])

    print("\n[Quantize] Building DataLoaders...")
    dataloaders = get_dataloaders()

    print("[Quantize] Building Model and loading trained checkpoint...")
    checkpoint_path = CONFIG.get("quantization", {}).get("checkpoint_path", "checkpoints/best_model.pt")
    model = get_model()
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    size_before = model_size_mb(model)
    print(f"[Quantize] Model size before quantization: {size_before:.2f} MB")

    print("[Quantize] Applying dynamic quantization (int8) to Linear layers...")
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    size_after = model_size_mb(quantized_model)
    print(f"[Quantize] Model size after quantization: {size_after:.2f} MB ({100 * size_after / size_before:.1f}% of original)")

    print("\n[Quantize] Starting testing of the quantized model (CPU)...")
    profiler = ResourceProfiler(interval=1.0)
    profiler.start()
    final_results = run_testing(model=quantized_model, test_loader=dataloaders["test"], device=DEVICE)
    profiler.stop()

    hardware_metrics = profiler.get_results()
    hardware_metrics["model_size_mb"] = size_after
    hardware_metrics["quantized"] = True

    save_to_csv(CONFIG, final_results, hardware_metrics)
    log_to_mlflow(CONFIG, final_results, hardware_metrics)

if __name__ == "__main__":
    main()

# python -m scoring.quantize
