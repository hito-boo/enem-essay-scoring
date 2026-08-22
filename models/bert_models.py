from config import CONFIG
from transformers import BertModel
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
        self.bert = BertModel.from_pretrained(bert_name)
        self.heads = heads

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = bert_out.pooler_output
        cls_repr = self.dropout(cls_repr)
        return [head(cls_repr) for head in self.heads]