from config import CONFIG
from models.bert_models import BaseModel
from models.competence_models import LinearHead, MultiLayerHead
from task_strategy import get_task_type, num_outputs_for_task
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
    num_outputs = num_outputs_for_task(get_task_type())
    for comp in range(1, 6):
        heads.append(competence_mapping[CONFIG["transformers"][f"c{comp}"]](
            dropout=CONFIG["params"]["drop_out"][comp],
            num_outputs=num_outputs,
        ))
    return nn.ModuleList(heads)