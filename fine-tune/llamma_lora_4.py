## Import and basic setup
import transformers, inspect, sys
import torch
from transformers import LlamaTokenizer, LlamaForSequenceClassification, Trainer, TrainingArguments

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset, DatasetDict
import evaluate

import pandas as pd
import numpy as np
import random
import sentencepiece as spm

from sklearn.model_selection import train_test_split

df = pd.read_pickle("/home/pingsong/OOD_TTA/data/raw/NewsTweets_ID.pkl")


# 1. Prepare toy dataset
def build_toy_dataset():
    texts = df['original_text'].tolist()
    labels = df['label'].tolist()
    
    ds = Dataset.from_dict({'text': texts, 'label': labels})
    ds = ds.train_test_split(test_size=0.2, seed=42)
    return DatasetDict({'train': ds['train'], 'test': ds['test']})


# 2. Load tokenizer & model using Llama classes
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
# Define pad token for batching
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = LlamaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
    load_in_8bit=True,
    device_map="auto"
)
# Align model pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# 3. Prepare model for LoRA + k-bit training
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)
model = get_peft_model(model, lora_config)

# 4. Tokenization function
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        max_length=128,
        truncation=True
    )

# 5. Prepare datasets
datasets = build_toy_dataset()
datasets = datasets.map(tokenize, batched=True)
datasets = datasets.rename_column("label", "labels")
datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./lora_sentiment",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    learning_rate=1e-4,
    do_eval=True,
    eval_steps=10,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=10,
    fp16=True,
    remove_unused_columns=False,
)

# 7. Trainer initialization
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = (preds == torch.tensor(labels)).float().mean()
    return {"accuracy": accuracy.item()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# 8. Fine-tune
trainer.train()

# 9. Save LoRA adapters
model.save_pretrained("./lora_news_adapter")
tokenizer.save_pretrained("lora-output/news_adapter/")
