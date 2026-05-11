import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
#from torch.utils.data import DataLoader
#from transformers import BitsAndBytesConfig

# Load Data
df_id = pd.read_pickle("/home/pingsong/OOD_TTA/data/raw/Sentiment_ID.pkl") 

df_ood = pd.read_pickle("/home/pingsong/OOD_TTA/augmentation/Sentiment/SE/Semeval.pkl") 
df_ood.insert(1, "augmentation", [''] * len(df_ood))



# Updated to Llama-3.1-8B (Base Model)
model_id = "meta-llama/Llama-3.1-8B"

# Note: Llama-3.1 is a gated model. You must have accepted Meta's terms 
# on Hugging Face and be logged in via `huggingface-cli login` in your terminal.
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # CRITICAL for batch generation

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, 
    #low_cpu_mem_usage=True,
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
- Transform the input sentence to an equivalent expression in the domain of Amazon reviews.
- Please match the style, tone, expression, vocabularies, sentence structure, and target objects of the provided example sentences from Amazon reviews, while maintaining equivalent semantic meaning.
- You may adapt the content and change details to sound natural for an Amazon review, as long as the overall sentiment category and intensity remain unchanged.

### Example Sentences and their domain: ###
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

### Input Sentence: 
Now rewrite ```"{input_sentence}"``` to match the style, tone, expression, vocabularies, sentence structure, and target objects of the provided example sentences from Amazon reviews, while maintaining equivalent semantic meaning.
You may adapt the content and change details to sound natural for an Amazon review, as long as the overall sentiment category and intensity remain unchanged.     

Return the text in the format: ```Paraphrased Text```

### Transformed Text ###
Paraphrased Text:
"""
    return prompt


np.random.seed(42)


def augmentation_hf_batched(df_id, df_ood, batch_size=8):
    all_prompts = []
    
    print(f"Preparing 80,000 prompts...")
    for _, row in df_ood.iterrows():
        for _ in range(4):
            examples = balanced_sample(df_id, n_samples=16, label_col='label')
            all_prompts.append(prompt_gen(examples, row["original_text"]))

    all_cleaned_outputs = []
    
    # Processing in batches to avoid VRAM explosion
    print(f"Starting batched inference (Batch Size: {batch_size})...")
    for i in range(0, len(all_prompts), batch_size):
        batch_texts = all_prompts[i : i + batch_size]
        
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=4096
        ).to(model.device)
        
        input_length = inputs.input_ids.shape[-1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                top_p=0.9,
                max_new_tokens=256,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        batch_decoded = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
        
        for text in batch_decoded:
            all_cleaned_outputs.append(clean_output(text))
            
        if i % (batch_size * 20) == 0:
            print(f"Progress: {i}/{len(all_prompts)} prompts generated")

    # Reshape back to list of 4 per row
    final_results = [all_cleaned_outputs[i : i + 4] for i in range(0, len(all_cleaned_outputs), 4)]
    df_ood['augmentation'] = final_results
    
    return df_ood

df_ood = augmentation_hf_batched(df_id, df_ood, batch_size=8)

df_ood["augmentation"] = df_ood["augmentation"].apply(lambda lst: [s.strip() for s in lst])

# Output filename reflects Llama usage
df_ood.to_pickle('/home/pingsong/OOD_TTA/augmentation_Llama3/SE_BR_llama3_base.pkl')



