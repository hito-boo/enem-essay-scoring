try:
    import torch.distributed.tensor
except ImportError:
    pass

from config import CONFIG
from peft import LoraConfig, inject_adapter_in_model

def apply_lora(bert):
    lora_config = CONFIG.get("lora", {})

    if not lora_config.get("enabled", False):
        return bert

    peft_config = LoraConfig(
        r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("alpha", 16),
        lora_dropout=lora_config.get("dropout", 0.1),
        target_modules=lora_config.get("target_modules", ["query", "value"]),
        bias="none",
    )

    bert = inject_adapter_in_model(peft_config, bert)

    total = sum(p.numel() for p in bert.parameters())
    trainable = sum(p.numel() for p in bert.parameters() if p.requires_grad)
    print(
        f"[LoRA] Enabled (r={peft_config.r}, alpha={peft_config.lora_alpha}, "
        f"target_modules={peft_config.target_modules}) - "
        f"Trainable params in body: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )

    return bert
