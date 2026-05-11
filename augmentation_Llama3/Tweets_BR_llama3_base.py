import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# Load Data
df_id = pd.read_pickle("/home/pingsong/OOD_TTA/data/raw/NewsTweets_ID.pkl") 

df_ood = pd.read_pickle("/home/pingsong/OOD_TTA/data/raw/Tweets.pkl") 
df_ood.insert(1, "augmentation", [''] * len(df_ood))

# Updated to Llama-3.1-8B (Base Model)
model_id = "meta-llama/Llama-3.1-8B"

# Note: Llama-3.1 is a gated model. You must have accepted Meta's terms 
# on Hugging Face and be logged in via `huggingface-cli login` in your terminal.
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Llama models typically do not have a pad_token defined out of the box.
# We set it to the eos_token to prevent errors during generation.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    device_map="auto" 
)

model.eval()

def clean_output(text):
    """
    Cleans up the output. Base models might try to continue the prompt format.
    """
    cleaned = re.sub(r'^paraphrased text:\s*', '', text, flags=re.IGNORECASE)
    
    # Truncate anything that looks like it's starting a new prompt block.
    stop_pattern = r"(###|Instruction|Input Sentence|Rewrite|17\.)"
    match = re.search(stop_pattern, cleaned, flags=re.IGNORECASE)
    if match:
         cleaned = cleaned[:match.start()]
            
    return cleaned.strip()

def query_Llama(prompt, temperature=0.7, max_tokens=256):
    # Feed the raw string directly to the base model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    input_length = inputs.input_ids.shape[-1]
    
    max_context = 4096 
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
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    return tokenizer.decode(output[0][input_length:], skip_special_tokens=True)

def balanced_sample(df, label_col, n_samples, random_state=None):
    unique_labels = df[label_col].unique()
    n_labels = len(unique_labels)
    
    base_count = n_samples // n_labels  
    remainder = n_samples % n_labels      
    
    sampled_list = []
    
    for label in unique_labels:
        df_label = df[df[label_col] == label]
        count = min(base_count, len(df_label))
        sampled = df_label.sample(n=count, random_state=random_state)
        sampled_list.append(sampled)
    
    extra_labels = np.random.RandomState(random_state).choice(unique_labels, size=remainder, replace=False)
    already_sampled = pd.concat(sampled_list).index
    
    for label in extra_labels:
        df_label = df[(df[label_col] == label) & (~df.index.isin(already_sampled))]
        if not df_label.empty:
            extra_sample = df_label.sample(n=1, random_state=random_state)
            sampled_list.append(extra_sample)
    
    sampled_df = pd.concat(sampled_list).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return sampled_df['original_text'].tolist()

def prompt_gen(example_sentences, input_sentence):
    prompt = f"""
### Instructions: ###
Rewrite the input text into the AG News domain, adjusting only background features such as writing structure, sentence flow, and discourse style.

- Preserve the original news topic (World, Sports, Business, or Sci/Tech) and factual meaning.
- Do not introduce new facts or opinions.
- Rewrite only the background style to match formal, neutral, third-person news reporting.
- Remove informal or social-media-specific language (e.g., slang, emojis, hashtags, first-person phrasing).

The output should resemble a short AG News article or headline-style paragraph.

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
Now rewrite ```"{input_sentence}"``` 

Return the text in the format: ```Paraphrased Text```

### Trnasformed Text ###
Paraphrased Text:
            """
    return prompt


np.random.seed(42)

def augmentation(df_id, df_ood):
    df_ood['augmentation'] = df_ood['augmentation'].astype(object)

    for index, row in df_ood.iterrows():
        tem_aug = []
        
        for i in range(4):
            example_sentences = balanced_sample(df_id, n_samples=16, label_col='label', random_state=None) 
            input_sentence = row["original_text"]
            prompt = prompt_gen(example_sentences, input_sentence)
            
            raw_aug = query_Llama(prompt)
            cleaned_aug = clean_output(raw_aug)
            tem_aug.append(cleaned_aug)

        df_ood.at[index, 'augmentation'] = tem_aug 

        if index % 50 == 0:
            print(f"Processed {index} rows")

    return df_ood

df_ood = augmentation(df_id, df_ood)

df_ood["augmentation"] = df_ood["augmentation"].apply(lambda lst: [s.strip() for s in lst])

# Output filename reflects Llama usage
df_ood.to_pickle('/home/pingsong/OOD_TTA/augmentation_Llama3/Tweets_BR_llama3_base.pkl')