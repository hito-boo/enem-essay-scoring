from config import CONFIG
from dataset.tokenizer import get_tokenized_dataset
from torch.utils.data import DataLoader

def get_dataloaders():
    tokenized_dataset = get_tokenized_dataset()

    dataloaders = {}
    dataloaders["train"] = DataLoader(tokenized_dataset["train"], batch_size=CONFIG["params"]["batch_size"], shuffle=True)
    dataloaders["validation"] = DataLoader(tokenized_dataset["validation"], batch_size=CONFIG["params"]["batch_size"], shuffle=False)
    dataloaders["test"] = DataLoader(tokenized_dataset["test"], batch_size=CONFIG["params"]["batch_size"], shuffle=False)

    return dataloaders