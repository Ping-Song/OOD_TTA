import os
import re
from dataclasses import dataclass
from typing import List, Callable, Tuple, Union

import numpy as np
import pandas as pd
import torch

# Transformers imports
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    LlamaTokenizer,
    LlamaForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig


# -----------------------------
# Utility: Device
# -----------------------------
def get_default_device(device_id: int | None = None):
    """
    Returns a torch.device.
    If device_id is given and valid, returns cuda:<device_id>,
    otherwise returns default cuda or cpu.
    """
    if torch.cuda.is_available():
        if device_id is not None:
            n = torch.cuda.device_count()
            if 0 <= device_id < n:
                return torch.device(f"cuda:{device_id}")
            else:
                print(f"[WARN] Requested device_id={device_id}, "
                      f"but only {n} CUDA devices available. Falling back to default CUDA.")
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------
# Utility: T5 Label Parsing
# -----------------------------
def parse_label(x: str, default: int = 0) -> int:
    """
    Try to extract an integer label from the decoded model output.
    Falls back to `default` if none is found.
    """
    if x is None:
        print("Warning: prediction is None. Using default:", default)
        return default

    x = x.strip()
    if x == "":
        print("Warning: empty prediction string. Using default:", default)
        return default

    # Look for integer in string
    m = re.search(r"-?\d+", x)
    if m:
        return int(m.group())

    print(f"Warning: non-numeric prediction: {x!r}. Using default={default}.")
    return default


# -----------------------------
# Base Interface
# -----------------------------
class BaseTextClassifier:
    """Simple interface that all models implement."""

    def predict(self, texts: List[str], batch_size: int = 32) -> List[int]:
        raise NotImplementedError

    def __call__(self, texts: List[str], batch_size: int = 32) -> List[int]:
        return self.predict(texts, batch_size=batch_size)


# ======================================================
# T5 MODELS (Seq2Seq)
# ======================================================

@dataclass
class T5SentimentModel(BaseTextClassifier):
    """Wrapper for Kyle1668/boss-sentiment-t5-large (3-class)."""
    model_name: str = "Kyle1668/boss-sentiment-t5-large"
    device_id: int = 0
    max_new_tokens: int = 5

    def __post_init__(self):
        self.device = get_default_device(self.device_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def predict(self, texts: List[str], batch_size: int = 32) -> List[int]:
        preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                output_sequences = self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=self.max_new_tokens,
                )
            
            decoded = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            preds.extend([parse_label(x) for x in decoded])
        return preds


@dataclass
class T5ToxicityModel(BaseTextClassifier):
    """Wrapper for Kyle1668/boss-toxicity-t5-large (Binary)."""
    model_name: str = "Kyle1668/boss-toxicity-t5-large"
    device_id: int = 0
    max_new_tokens: int = 5

    def __post_init__(self):
        self.device = get_default_device(self.device_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def predict(self, texts: List[str], batch_size: int = 32) -> List[int]:
        preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                output_sequences = self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=self.max_new_tokens,
                )
            decoded = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            preds.extend([parse_label(x) for x in decoded])
        return preds


@dataclass
class T5AGNewsModel(BaseTextClassifier):
    """Wrapper for Kyle1668/ag-news-t5-large (Topic Classification)."""
    model_name: str = "Kyle1668/ag-news-t5-large"
    device_id: int = 0
    max_new_tokens: int = 5 

    def __post_init__(self):
        self.device = get_default_device(self.device_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def predict(self, texts: List[str], batch_size: int = 32) -> List[int]:
        preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                output_sequences = self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=self.max_new_tokens,
                )
            decoded = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            # Warning: Ensure this model outputs NUMBERS (e.g. "1") and not WORDS (e.g. "Sports")
            preds.extend([parse_label(x) for x in decoded])
        return preds


# ======================================================
# BERT MODELS (Sequence Classification)
# ======================================================

@dataclass
class BertSentimentModel(BaseTextClassifier):
    """Wrapper for Kyle1668/boss-sentiment-bert-base-uncased."""
    model_name: str = "Kyle1668/boss-sentiment-bert-base-uncased"
    device_id: int = 0

    def __post_init__(self):
        self.device = get_default_device(self.device_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def predict(self, texts: List[str], batch_size: int = 32) -> List[int]:
        preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
                batch_pred = logits.argmax(dim=-1).cpu().numpy().tolist()
            preds.extend(batch_pred)
        return preds


@dataclass
class BertToxicityModel(BaseTextClassifier):
    """Wrapper for Kyle1668/boss-toxicity-bert-base-uncased."""
    model_name: str = "Kyle1668/boss-toxicity-bert-base-uncased"
    device_id: int = 0

    def __post_init__(self):
        self.device = get_default_device(self.device_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def predict(self, texts: List[str], batch_size: int = 32) -> List[int]:
        preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
                batch_pred = logits.argmax(dim=-1).cpu().numpy().tolist()
            preds.extend(batch_pred)
        return preds


@dataclass
class BertAGNewsModel(BaseTextClassifier):
    """Wrapper for Kyle1668/ag-news-bert-base-uncased."""
    model_name: str = "Kyle1668/ag-news-bert-base-uncased"
    device_id: int = 0
    max_length: int = 512

    def __post_init__(self):
        self.device = get_default_device(self.device_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def predict(self, texts: List[str], batch_size: int = 32) -> List[int]:
        preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
                batch_preds = torch.argmax(logits, dim=-1)
            preds.extend(batch_preds.cpu().tolist())
        return preds


# ======================================================
# LLaMA MODELS (LoRA)
# ======================================================

@dataclass
class LlamaSentimentModel(BaseTextClassifier):
    """LLaMA-2 + LoRA sentiment classifier (3 labels)."""
    
    # ADJUST PATHS AS NEEDED
    lora_path: str = "/home/pingsong/OOD_TTA/fine-tune/lora_sentiment_adapter"
    tokenizer_path: str = "/home/pingsong/OOD_TTA/fine-tune/sentiment_adapter"
    num_labels: int = 3
    gpu_id: int | None = None

    def __post_init__(self):
        if self.gpu_id is not None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.gpu_id}")
        else:
            self.device = get_default_device()

        if not os.path.isdir(self.lora_path):
            raise FileNotFoundError(f"LlamaSentimentModel: lora_path '{self.lora_path}' does not exist.")
        if not os.path.isdir(self.tokenizer_path):
            raise FileNotFoundError(f"LlamaSentimentModel: tokenizer_path '{self.tokenizer_path}' does not exist.")

        peft_config = PeftConfig.from_pretrained(self.lora_path)
        base_model = LlamaForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=self.num_labels,
            torch_dtype=torch.float16,
        )
        base_model.to(self.device)

        self.model = PeftModel.from_pretrained(base_model, self.lora_path, torch_dtype=torch.float16)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def predict(self, texts: List[str], batch_size: int = 32) -> List[int]:
        preds = []
        device = self.device
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
                batch_pred = logits.argmax(dim=-1).cpu().tolist()
            preds.extend(batch_pred)
        return preds


@dataclass
class LlamaToxicityModel(BaseTextClassifier):
    """LLaMA-2 + LoRA toxicity classifier (Binary, 8-bit)."""
    
    base_model: str = "meta-llama/Llama-2-7b-hf"
    # ADJUST PATHS AS NEEDED
    lora_path: str = "/home/pingsong/OOD_TTA/fine-tune/lora_binary_adapter"
    tokenizer_path: str | None = None
    num_labels: int = 2
    gpu_id: int | None = None
    max_length: int = 512

    def __post_init__(self):
        if os.path.isdir(self.lora_path) is False and not self.lora_path.startswith(("meta-", "hf-", "org/", "username/")):
            raise FileNotFoundError(f"LlamaToxicityModel: lora_path '{self.lora_path}' does not exist.")

        # Logic for 8-bit loading with device_map
        if torch.cuda.is_available() and self.gpu_id is not None:
            device_map = {"": self.gpu_id}
        else:
            device_map = "auto"

        base_model = LlamaForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=self.num_labels,
            torch_dtype=torch.float16, # <-- Replaced 8-bit with native 16-bit
            device_map=device_map,
        )

        self.model = PeftModel.from_pretrained(base_model, self.lora_path)
        self.model.eval()
        self.device = next(self.model.parameters()).device # Get actual device

        tokenizer_src = self.tokenizer_path or self.base_model
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_src)
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def predict(self, texts: List[str], batch_size: int = 32) -> List[int]:
        preds = []
        # Input normalization
        if isinstance(texts, str):
            texts = [texts]

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length", # Note: using max_length padding here as per your original code
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
                batch_pred = logits.argmax(dim=-1).cpu().tolist()
            preds.extend(batch_pred)
        return preds

# ======================================================
# LLaMA MODELS (LoRA) - AG News Topic Classification
# ======================================================

@dataclass
class LlamaAGNewsModel(BaseTextClassifier):
    """
    LLaMA + LoRA news topic classifier (AG News: 4 labels).

    Expects a LoRA adapter saved via:
        model.save_pretrained("./lora_toxicity_adapter_4")
    and tokenizer saved via:
        tokenizer.save_pretrained("lora-output/toxicity_adapter_4/")
    """

    # Your saved adapter/tokenizer paths
    lora_path: str = "./lora_news_adapter"
    tokenizer_path: str = "lora-output/news_adapter"

    # AG News has 4 labels
    num_labels: int = 4

    # runtime
    gpu_id: int | None = None
    max_length: int = 512
    load_in_8bit: bool = False # set False if you want fp16 loading

    def __post_init__(self):
        # Validate paths
        if not os.path.isdir(self.lora_path):
            raise FileNotFoundError(
                f"LlamaAGNewsModel: lora_path '{self.lora_path}' does not exist."
            )
        if not os.path.isdir(self.tokenizer_path):
            raise FileNotFoundError(
                f"LlamaAGNewsModel: tokenizer_path '{self.tokenizer_path}' does not exist."
            )

        # device_map logic (consistent with your LlamaToxicityModel)
        if torch.cuda.is_available() and self.gpu_id is not None:
            device_map = {"": self.gpu_id}
        else:
            device_map = "auto"

        # Read base model name from adapter config
        peft_config = PeftConfig.from_pretrained(self.lora_path)
        base_model_name = peft_config.base_model_name_or_path

        # Load base model (optionally 8-bit)
        base_kwargs = dict(
            num_labels=self.num_labels,
            device_map=device_map,
        )

        if self.load_in_8bit:
            base_kwargs["load_in_8bit"] = True
        else:
            base_kwargs["torch_dtype"] = torch.float16

        base_model = LlamaForSequenceClassification.from_pretrained(
            base_model_name,
            **base_kwargs,
        )

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, self.lora_path)
        self.model.eval()

        # Actual device (important when device_map="auto")
        self.device = next(self.model.parameters()).device

        # Tokenizer from your saved tokenizer path
        self.tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def predict(self, texts: List[str], batch_size: int = 32) -> List[int]:
        # Normalize input
        if isinstance(texts, str):
            texts = [texts]

        preds: List[int] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                logits = self.model(**enc).logits
                batch_pred = logits.argmax(dim=-1).cpu().tolist()

            preds.extend(batch_pred)

        return preds
# -----------------------------
# TTA / Dataframe Logic
# -----------------------------
def tta_on_dataframe(
    df: pd.DataFrame,
    aug_col_name: str,
    predictor: Union[BaseTextClassifier, Callable[[List[str]], List[int]]],
    text_col: str = "original_text",
    label_col: str = "label",
    prefix: str = "",
    max_augs_for_vote: int | None = None,
    compute_original: bool = True,
    compute_tta: bool = True,
) -> Tuple[pd.DataFrame, float | None, float | None]:
    
    df = df.copy()
    df[label_col] = df[label_col].astype(int)

    if isinstance(predictor, BaseTextClassifier):
        pred_fn = predictor.predict
    else:
        pred_fn = predictor

    # 0. Sanity check + normalize augmentation column
    if compute_tta:
        if aug_col_name not in df.columns:
            raise ValueError(f"Augmentation column '{aug_col_name}' not in DataFrame.")

        def normalize_aug_cell(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return None
            if isinstance(x, list):
                return x
            if isinstance(x, (tuple, np.ndarray)):
                return list(x)
            if isinstance(x, str):
                return x
            return None

        df[aug_col_name] = df[aug_col_name].apply(normalize_aug_cell)

    # 1. Original predictions
    orig_col = f"{prefix}original_prediction"
    if orig_col not in df.columns:
        print(f"Computing original predictions ({prefix.rstrip('_') or 'model'})...")
        df[orig_col] = pred_fn(df[text_col].tolist())

    original_acc: float | None = None
    if compute_original:
        original_acc = (df[orig_col] == df[label_col]).mean()
        print(f"Original accuracy ({aug_col_name}): {original_acc:.4f}")

    # 2. TTA predictions
    tta_acc: float | None = None
    if compute_tta:
        gen_col = f"{prefix}gen_{aug_col_name}"
        print(f"Running TTA for column: {aug_col_name} (prefix='{prefix}')")

        all_tta_preds: list[list[int]] = []
        for _, aug_list in df[aug_col_name].items():
            if aug_list is None or (isinstance(aug_list, float) and pd.isna(aug_list)):
                all_tta_preds.append([])
            elif isinstance(aug_list, str):
                preds = pred_fn([aug_list])
                all_tta_preds.append(preds)
            elif isinstance(aug_list, list) and len(aug_list) > 0:
                preds = pred_fn(aug_list)
                all_tta_preds.append(preds)
            else:
                all_tta_preds.append([])

        df[gen_col] = all_tta_preds

        # Vote Logic
        gen_with_orig_col = f"{gen_col}_with_orig"

        def build_votes(orig, aug_preds):
            if max_augs_for_vote is None:
                chosen_augs = aug_preds
            else:
                k = max_augs_for_vote
                chosen_augs = aug_preds[:k]
            return [orig] + chosen_augs

        df[gen_with_orig_col] = [
            build_votes(orig, aug_preds)
            for orig, aug_preds in zip(df[orig_col], df[gen_col])
        ]

        def majority_vote(votes: list[int]) -> int:
            votes = np.asarray(votes, dtype=int)
            if votes.size == 0:
                return 0
            counts = np.bincount(votes, minlength=votes.max() + 1)
            return int(counts.argmax())

        pred_col = f"{prefix}{aug_col_name}_prediction"
        df[pred_col] = df[gen_with_orig_col].apply(majority_vote)

        tta_acc = (df[pred_col] == df[label_col]).mean()
        print(
            f"TTA ({aug_col_name}) accuracy ({prefix.rstrip('_') or 'model'}) "
            f"with max_augs_for_vote={max_augs_for_vote}: {tta_acc:.4f}"
        )

    return df, original_acc, tta_acc