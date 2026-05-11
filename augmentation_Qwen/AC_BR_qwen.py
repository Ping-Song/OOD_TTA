import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, pipeline, RobertaModel, AutoModelForSequenceClassification, TrainingArguments, Trainer, BertForSequenceClassification, AutoModelForCausalLM
from torch.utils.data import Dataset
import openai
import os
import random
import sentencepiece as spm
import re

df_id = pd.read_pickle("/home/pingsong/OOD_TTA/data/raw/Toxicity_ID.pkl")

df_ood= pd.read_pickle("/home/pingsong/OOD_TTA/data/raw/AdvCivil.pkl")
df_ood.insert(1, "augmentation", [''] * len(df_ood))

device = "cuda" if torch.cuda.is_available() else "cpu"

# Updated to Qwen2.5-7B-Instruct
model_id = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # bfloat16 or float16 is strongly recommended for Qwen over float32
    low_cpu_mem_usage=True,
    device_map="auto" # Let accelerate handle optimal device placement automatically
)

model.eval()

def clean_output(text):
    """
    Removes the 'Paraphrased Text:' prefix and any leading/trailing whitespace.
    """
    # This regex handles variations in case and spacing
    cleaned = re.sub(r'^paraphrased text:\s*', '', text, flags=re.IGNORECASE)
    return cleaned.strip()

def query_Qwen(prompt, temperature=0.7, max_tokens=256):
    # Format the prompt using Qwen's native chat structure for best results
    messages = [
        {"role": "system", "content": "You are a helpful assistant that accurately paraphrases text."},
        {"role": "user", "content": prompt}
    ]
    
    # Apply the chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    input_length = inputs.input_ids.shape[-1]
    
    # Qwen2.5 actually supports up to 128k context, but we will keep your 4096 safety limit
    max_context = 4096 

    # ensure we don't go past context window
    safe_max_new = min(max_tokens, max_context - input_length)
    if safe_max_new <= 0:
        safe_max_new = 1 

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            top_p=0.95,
            top_k=0,
            max_new_tokens=safe_max_new,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output[0][input_length:], skip_special_tokens=True)

def balanced_sample(df, label_col, n_samples, random_state=None):
    # Get unique labels and their count
    unique_labels = df[label_col].unique()
    n_labels = len(unique_labels)
    
    # Determine the base number of samples per label and the remainder
    base_count = n_samples // n_labels  
    remainder = n_samples % n_labels      
    
    sampled_list = []
    
    # Sample base_count examples from each label
    for label in unique_labels:
        df_label = df[df[label_col] == label]
        # If a group has fewer than base_count examples, sample as many as available
        count = min(base_count, len(df_label))
        sampled = df_label.sample(n=count, random_state=random_state)
        sampled_list.append(sampled)
    
    # For the remaining samples, randomly choose labels and sample one extra from each
    extra_labels = np.random.RandomState(random_state).choice(unique_labels, size=remainder, replace=False)
    # To avoid duplicating indices, combine the already-sampled indices
    already_sampled = pd.concat(sampled_list).index
    for label in extra_labels:
        df_label = df[(df[label_col] == label) & (~df.index.isin(already_sampled))]
        # Make sure there is at least one example left to sample
        if not df_label.empty:
            extra_sample = df_label.sample(n=1, random_state=random_state)
            sampled_list.append(extra_sample)
    
    # Combine and shuffle the final sample
    sampled_df = pd.concat(sampled_list).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return sampled_df['original_text'].tolist()

def prompt_gen(example_sentences, input_sentence):
    prompt = f"""
            ### Instructions ###
            Rewrite the input sentence into the Civil Comments domain, 
            adjusting only background features such as writing structure, 
            sentence flow, and discourse style.

            CRITICALLY:
            - Preserve all toxic, abusive, insulting, or hateful expressions EXACTLY.
            - Do NOT soften, weaken, sanitize, or remove any toxic or profane words.
            - Do NOT replace slurs or insults with milder alternatives.
            - Do NOT change the target of the toxicity.
            - Do NOT add meta-language about toxicity (e.g., "this comment is offensive").
            - Only modify non-semantic stylistic aspects (punctuation, connective words, flow, formatting).

            ###Example Sentences and their domain:###
             1. {example_sentences[0]};
             2. {example_sentences[1]};
             3. {example_sentences[2]};
             4. {example_sentences[3]};
             5. {example_sentences[4]};
             6. {example_sentences[5]};
             7. {example_sentences[6]};
             8. {example_sentences[7]};
             9. {example_sentences[8]};
             10. {example_sentences[9]};
             11. {example_sentences[10]};
             12. {example_sentences[11]};
             13. {example_sentences[12]};
             14. {example_sentences[13]};
             15. {example_sentences[14]};
             16. {example_sentences[15]}

            ###Input Sentence: 
            {input_sentence};              
    
            ### Trnasformed Text ###
            Rewrite the sentence in Civil Comments style while keeping all toxic expressions verbatim and maintaining the same level of toxicity. Only shift background writing style.
        """
    return prompt

np.random.seed(42)

def augmentation(df_id, df_ood):
    # CRITICAL: Ensure the column is of 'object' dtype so it can hold lists
    df_ood['augmentation'] = df_ood['augmentation'].astype(object)

    for index, row in df_ood.iterrows():
        tem_aug = []
        
        for i in range(4):
            example_sentences = balanced_sample(df_id, n_samples=16, label_col='label', random_state=None) 
            input_sentence = row["original_text"]
            prompt = prompt_gen(example_sentences, input_sentence)
            
            # Generate and clean
            raw_aug = query_Qwen(prompt)
            cleaned_aug = clean_output(raw_aug)
            tem_aug.append(cleaned_aug)

        # Use .at with index and column name
        # If this still complains, we use the "list-wrap" trick:
        df_ood.at[index, 'augmentation'] = tem_aug 

        if index % 50 == 0:
            print(f"Processed {index} rows")

    return df_ood

df_ood = augmentation(df_id, df_ood)

df_ood["augmentation"] = df_ood["augmentation"].apply(lambda lst: [s.strip() for s in lst])

# Updated output filename to reflect Qwen usage
df_ood.to_pickle('/home/pingsong/OOD_TTA/augmentation_Qwen/AC_BR_qwen.pkl')