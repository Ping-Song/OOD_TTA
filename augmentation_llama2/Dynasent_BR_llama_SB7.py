import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, pipeline, RobertaModel, AutoModelForSequenceClassification, TrainingArguments, Trainer, BertForSequenceClassification, AutoModelForCausalLM
from torch.utils.data import Dataset
import openai
import os
import random
import sentencepiece as spm

import random
import os

df_id = pd.read_pickle("/home/pingsong/OOD_TTA/data/raw/Sentiment_ID.pkl") 

df_ood = pd.read_pickle("/home/pingsong/OOD_TTA/data/raw/Dynasent.pkl") 
df_ood.insert(1, "augmentation", [''] * len(df_ood))


# Use a pipeline as a high-level helper
#pipe = pipeline("text-generation", model="stabilityai/StableBeluga-7B")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "stabilityai/StableBeluga-7B",
    use_fast=False
)

model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/StableBeluga-7B",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
).to(device)
    # ^^^ use the same device variable

model.eval()

"""
def query_SB7(prompt, temperature=0.7, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            top_p=0.95,
            top_k=0,
            max_new_tokens=max_tokens,
            temperature=temperature,
            eos_token_id = tokenizer.eos_token_id
        )
    input_length = inputs.input_ids.shape[-1]
    return tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
"""

def query_SB7(prompt, temperature=0.7, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    input_length = inputs.input_ids.shape[-1]
    max_context = 4095

    # ensure we don't go past context window
    safe_max_new = min(max_tokens, max_context - input_length)
    if safe_max_new <= 0:
        # prompt already at or near limit; either shorten or just not generate
        safe_max_new = 1  # or raise an error / return empty

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            top_p=0.95,
            top_k=0,
            max_new_tokens=safe_max_new,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output[0][input_length:], skip_special_tokens=True)

def balanced_sample(df, label_col, n_samples, random_state=None):
    # Get unique labels and their count
    unique_labels = df[label_col].unique()
    n_labels = len(unique_labels)
    
    # Determine the base number of samples per label and the remainder
    base_count = n_samples // n_labels  # e.g. 16//3 = 5
    remainder = n_samples % n_labels      # e.g. 16%3 = 1
    
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


def prompt_gen(example_sentences,input_sentence):
    prompt = f"""
### Instructions: ###
- The assistant is to transform the input sentence to an equivalent experssion in the domain of Amazon reviews.
- Please matche the style, tone, expression, vocabularies, sentence structure, target objects of the provided example sentences from Amazon review, while maintaining equivalent semantic meaning.
- You may adapt the content and change details to sound natural for an Amazon review, as long as the overall sentiment category and intensity remain unchanged.

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
Now rewrite ```"{input_sentence}"``` to matche the style, tone, expression, vocabularies, sentence structure, target objects of the provided example sentences from Amazon review, while maintaining equivalent semantic meaning.
You may adapt the content and change details to sound natural for an Amazon review, as long as the overall sentiment category and intensity remain unchanged.     

Return the text in the format: ```Paraphrased Text```

### Trnasformed Text ###
Paraphrased Text:
            """
    return prompt


np.random.seed(42)

def augmentation(df_id,df_ood):
    for index, row in df_ood.iterrows():
        tem_aug = []
        
        for i in range(4):
            example_sentences = balanced_sample(df_id, n_samples=16, label_col='label',  random_state=None) 
            #example_sentences, domain = balanced_sample(df_id, n_samples=16, label_col='label', domain_col='domain', random_state=42)

            # Test input
            input_sentence = row["original_text"]

            prompt = prompt_gen(example_sentences,input_sentence)
            
            aug = query_SB7(prompt)
            tem_aug.append(aug)

        df_ood.at[index, 'augmentation'] = tem_aug

        if index % 20 == 0:
            print(f"Processed {index} rows")

    return(df_ood)


df_ood = augmentation(df_id,df_ood)


df_ood["augmentation"] = df_ood["augmentation"].apply(lambda lst: [s.strip() for s in lst])

df_ood.to_pickle('/home/pingsong/OOD_TTA/augmentation/Sentiment/DY/DY_BR_llamaSB7_9_12.pkl')