from config import CONFIG
from transformers import AutoModel
from models.lora import apply_lora
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(
        self,
        heads, # nn.ModuleList([C1_Model, C2_Model, C3_Model, C4_Model, C5_Model])
        bert_name=CONFIG["transformers"]["bert"], # Ex. neuralmind/bert-base-portuguese-cased
        dropout=CONFIG["params"]["drop_out"][0], # Ex. 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.bert = AutoModel.from_pretrained(bert_name)
        self.bert = apply_lora(self.bert)
        self.heads = heads

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler = getattr(bert_out, "pooler_output", None)
        cls_repr = pooler if pooler is not None else bert_out.last_hidden_state[:, 0]
        cls_repr = self.dropout(cls_repr)
        return [head(cls_repr) for head in self.heads]