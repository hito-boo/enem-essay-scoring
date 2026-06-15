from config import CONFIG
from models.bert_models import BaseModel
from models.competence_models import LinearHead, MultiLayerHead
import torch.nn as nn

bert_mapping = {
    "base": BaseModel
}

competence_mapping = {
    "linear": LinearHead,
    "multi": MultiLayerHead,
}

def get_model():
    heads = get_heads()
    return bert_mapping[CONFIG["transformers"]["body"]](heads=heads)

def get_heads():
    heads = []
    for comp in range(1, 6):
        heads.append(competence_mapping[CONFIG["transformers"][f"c{comp}"]](dropout=CONFIG["params"]["drop_out"][comp]))
    return nn.ModuleList(heads)