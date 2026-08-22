import json
import os
import random
import torch
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

CONFIG_PATH = Path(os.environ.get("AES_CONFIG_PATH", ROOT_DIR / "aesconfig.json"))

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 21):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Config] Global Seed: {seed}")