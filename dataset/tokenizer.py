from config import CONFIG
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer"]) # Ex. neuralmind/bert-base-portuguese-cased

def get_tokenized_dataset():
    essays_dataset = load_dataset(
        CONFIG["dataset_corpus"]["link"], # Ex. kamel-usp/aes_enem_dataset
        name=CONFIG["dataset_corpus"]["name"] # Ex. JBCS2025
    )

    tokenized_dataset = essays_dataset.map(tokenize_and_prepare_grades, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "grades"])
    
    return tokenized_dataset

def tokenize_and_prepare_grades(batch):
    inputs = tokenizer(
        batch["essay_text"],
        padding = "max_length",
        truncation = True,
        max_length = 512
    )

    inputs["grades"] = [
        [
            int(grades[0] / 40),
            int(grades[1] / 40),
            int(grades[2] / 40),
            int(grades[3] / 40),
            int(grades[4] / 40),
            int(grades[5])
        ]
        for grades in batch["grades"]
    ]

    return inputs