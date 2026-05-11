import torch
import pandas as pd
import numpy as np
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import time
import openai


# Load API Key securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-...")

# Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)


# ==========================================
# 1. Define Your Prompt Library
# ==========================================
def get_prompt(prompt_type, example_sentences, input_sentence):
    # Some of your prompts used semicolons at the end of examples, some didn't. 
    # These two variables handle both cleanly.
    examples_str_clean = "\n".join([f"{i+1}. {ex}" for i, ex in enumerate(example_sentences)])
    examples_str_semi = "\n".join([f"{i+1}. {ex};" for i, ex in enumerate(example_sentences)])

    if prompt_type == "sentiment":
        return f"""### Instructions: ###
- Transform the input sentence to an equivalent expression in the domain of Amazon reviews.
- Please match the style, tone, expression, vocabularies, sentence structure, and target objects of the provided example sentences from Amazon reviews, while maintaining equivalent semantic meaning.
- You may adapt the content and change details to sound natural for an Amazon review, as long as the overall sentiment category and intensity remain unchanged.

### Example Sentences and their domain: ###
{examples_str_semi}

### Input Sentence: 
Now rewrite ```"{input_sentence}"``` to match the style, tone, expression, vocabularies, sentence structure, and target objects of the provided example sentences from Amazon reviews, while maintaining equivalent semantic meaning.
You may adapt the content and change details to sound natural for an Amazon review, as long as the overall sentiment category and intensity remain unchanged.     

Return the text in the format: ```Paraphrased Text```

### Transformed Text ###
Paraphrased Text:
"""

    elif prompt_type == "toxicity":
        return f"""### Instructions ###
Rewrite the input sentence into the Civil Comments domain, adjusting only background features such as writing structure, sentence flow, and discourse style.

CRITICALLY:
- Preserve all toxic, abusive, insulting, or hateful expressions EXACTLY.
- Do NOT soften, weaken, sanitize, or remove any toxic or profane words.
- Do NOT replace slurs or insults with milder alternatives.
- Do NOT change the target of the toxicity.
- Do NOT add meta-language about toxicity (e.g., "this comment is offensive").
- Only modify non-semantic stylistic aspects (punctuation, connective words, flow, formatting).

### Example Sentences and their domain: ###
{examples_str_clean}

### Input Sentence: 
{input_sentence}

### Transformed Text ###
"""

    elif prompt_type == "agnews":
        return f"""### Instructions: ###
Rewrite the input text into the AG News domain, adjusting only background features such as writing structure, sentence flow, and discourse style.

- Preserve the original news topic (World, Sports, Business, or Sci/Tech) and factual meaning.
- Do not introduce new facts or opinions.
- Rewrite only the background style to match formal, neutral, third-person news reporting.
- Remove informal or social-media-specific language (e.g., slang, emojis, hashtags, first-person phrasing).

The output should resemble a short AG News article or headline-style paragraph.

### Example Sentences and their domain: ###
{examples_str_semi}

### Input Sentence: 
Now rewrite ```"{input_sentence}"``` 

Return the text in the format: ```Paraphrased Text```

### Transformed Text ###
Paraphrased Text:
"""

    elif prompt_type == "icr":
        return f"""### Instructions ###
Paraphrase the input text as if it was one of the examples. Change the details of the text if necessary.

### Style Examples ###
{examples_str_clean}

### Input Text ###
Now paraphrase ```"{input_sentence}"``` as if it was one of the examples. Change the details of the text if necessary.

Return ONLY the paraphrased text. Do not include introductory conversational filler.
"""

    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
# ==========================================
# 2. Core Functions
# ==========================================
def clean_output(text):
    cleaned = re.sub(r'^paraphrased text:\s*', '', text, flags=re.IGNORECASE)
    stop_pattern = r"(###|Instruction|Input Sentence|Rewrite|17\.)"
    match = re.search(stop_pattern, cleaned, flags=re.IGNORECASE)
    if match:
         cleaned = cleaned[:match.start()]
    return cleaned.strip()


def query_model(model, tokenizer, prompt, model_type, temperature=0.7, max_tokens=256):
    start_time = time.perf_counter() # Start timer
    
    # Route for GPT-4o
    if model_type == "gpt-4o":
        completion = model.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at transforming sentences to other domains and shift semantic meaning to something in that domain."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        latency = time.perf_counter() - start_time
        in_tokens = completion.usage.prompt_tokens
        out_tokens = completion.usage.completion_tokens
        return completion.choices[0].message.content.strip(), latency, in_tokens, out_tokens

    # Route for local Hugging Face Models
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant that strictly follows instructions to transform text."},
        {"role": "user", "content": prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    input_length = inputs.input_ids.shape[-1]
    
    safe_max_new = max(1, min(max_tokens, 4096 - input_length))

    terminators = [tokenizer.eos_token_id]
    if model_type == "llama3":
        if "<|eot_id|>" in tokenizer.vocab:
            terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            top_p=0.95,
            top_k=0,
            max_new_tokens=safe_max_new,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=terminators,
        )

    latency = time.perf_counter() - start_time # End timer
    output_tokens = output[0][input_length:]
    num_out_tokens = len(output_tokens)
    
    decoded_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    
    return decoded_text, latency, input_length, num_out_tokens


def balanced_sample(df, label_col, n_samples, random_state=None):
    unique_labels = df[label_col].unique()
    n_labels = len(unique_labels)
    base_count = n_samples // n_labels  
    remainder = n_samples % n_labels      
    
    sampled_list = []
    for label in unique_labels:
        df_label = df[df[label_col] == label]
        count = min(base_count, len(df_label))
        sampled_list.append(df_label.sample(n=count, random_state=random_state))
    
    extra_labels = np.random.RandomState(random_state).choice(unique_labels, size=remainder, replace=False)
    already_sampled = pd.concat(sampled_list).index
    
    for label in extra_labels:
        df_label = df[(df[label_col] == label) & (~df.index.isin(already_sampled))]
        if not df_label.empty:
            sampled_list.append(df_label.sample(n=1, random_state=random_state))
    
    sampled_df = pd.concat(sampled_list).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return sampled_df['original_text'].tolist()

def augmentation(df_id, df_ood, model, tokenizer, prompt_type, model_type):
    df_ood['augmentation'] = df_ood['augmentation'].astype(object)

    # Initialize metric trackers
    metrics = {
        "total_latency": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_calls": 0
    }

    for index, row in df_ood.iterrows():
        tem_aug = []
        for i in range(4):
            example_sentences = balanced_sample(df_id, n_samples=16, label_col='label') 
            input_sentence = row["original_text"]
            
            prompt = get_prompt(prompt_type, example_sentences, input_sentence)
            
            # Unpack the new metrics
            raw_aug, lat, in_tok, out_tok = query_model(model, tokenizer, prompt, model_type)
            tem_aug.append(clean_output(raw_aug))
            
            # Update metrics
            metrics["total_latency"] += lat
            metrics["total_input_tokens"] += in_tok
            metrics["total_output_tokens"] += out_tok
            metrics["total_calls"] += 1

        df_ood.at[index, 'augmentation'] = tem_aug 

        if index % 50 == 0 and index > 0:
            print(f"Processed {index} rows...")

    return df_ood, metrics
# ==========================================
# 3. Main Execution Block
# ==========================================
# ==========================================
# 3. Main Execution Block
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Run LLM Data Augmentation")
    parser.add_argument("--id_data", type=str, required=True, help="Path to In-Domain dataset")
    parser.add_argument("--ood_data", type=str, required=True, help="Path to Out-Of-Domain dataset")
    parser.add_argument("--prompt_type", type=str, choices=["sentiment", "toxicity", "agnews", "icr"], required=True, help="Prompt template to use")
    parser.add_argument("--augmentation_model", type=str, choices=["llama3", "qwen", "gpt-4o"], required=True, help="Model to use: 'llama3', 'qwen' or 'gpt-4o'")
    parser.add_argument("--output_file", type=str, default=None, help="Optional: Force a specific output path")
    
    # NEW: Add an optional sample size argument
    parser.add_argument("--sample_size", type=int, default=None, help="Number of rows to sample for quick latency testing")
    
    args = parser.parse_args()
    np.random.seed(42)

    # Automatically construct output filename if not provided
    if args.output_file is None:
        dataset_name = os.path.splitext(os.path.basename(args.ood_data))[0]
        output_dir = f"/home/pingsong/OOD_TTA/augmentation_{args.augmentation_model}"
        os.makedirs(output_dir, exist_ok=True)
        # Append '_sample' to the filename so you don't overwrite your full dataset runs
        suffix = f"_sample{args.sample_size}" if args.sample_size else ""
        out_path = os.path.join(output_dir, f"{dataset_name}_{args.prompt_type}{suffix}.pkl")
    else:
        out_path = args.output_file

    print("Loading data...")
    df_id = pd.read_pickle(args.id_data)
    df_ood = pd.read_pickle(args.ood_data)
    
    # NEW: Apply sampling if requested
    if args.sample_size is not None:
        print(f"Sampling {args.sample_size} rows for testing...")
        # Use min() to prevent errors if sample_size > dataset size
        safe_sample = min(args.sample_size, len(df_ood))
        df_ood = df_ood.sample(n=safe_sample, random_state=42).reset_index(drop=True)

    df_ood.insert(1, "augmentation", pd.Series([[] for _ in range(len(df_ood))], dtype=object))

    # Model Selection Logic
    if args.augmentation_model == "llama3":
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
    elif args.augmentation_model == "qwen":
        model_id = "Qwen/Qwen2.5-7B-Instruct"

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if args.augmentation_model == "qwen" else torch.bfloat16, 
        low_cpu_mem_usage=True,
        device_map="auto" 
    )
    model.eval()

    print(f"Starting augmentation with '{args.prompt_type}' prompt using {args.augmentation_model}...")
    
    # Call the updated augmentation function that returns metrics
    df_ood, metrics = augmentation(df_id, df_ood, model, tokenizer, args.prompt_type, args.augmentation_model)

    df_ood["augmentation"] = df_ood["augmentation"].apply(lambda lst: [s.strip() for s in lst])
    
    # --- CALCULATE AND PRINT LATENCY/FLOPs TABLE ---
    calls = metrics["total_calls"]
    if calls > 0:
        avg_lat = metrics["total_latency"] / calls
        avg_in = metrics["total_input_tokens"] / calls
        avg_out = metrics["total_output_tokens"] / calls
        
        params = 8e9 if args.augmentation_model == "llama3" else 7e9
        avg_flops = 2 * params * (avg_in + avg_out)
        
        print("\n" + "="*50)
        print("📊 LATENCY & RESOURCE ANALYSIS (PER AUGMENTATION)")
        print("="*50)
        print(f"Sample Size      : {len(df_ood)} rows ({calls} model calls)")
        print(f"Prompt Type      : {args.prompt_type.upper()}")
        print(f"Model            : {args.augmentation_model}")
        print(f"Avg Wall-Clock   : {avg_lat:.2f} seconds")
        print(f"Avg Input Tokens : {avg_in:.1f}")
        print(f"Avg Output Tokens: {avg_out:.1f}")
        if args.augmentation_model != "gpt-4o":
             print(f"Est. TFLOPs      : {avg_flops / 1e12:.3f} TFLOPs")
        print("="*50 + "\n")

    print(f"Saving to {out_path}...")
    df_ood.to_pickle(out_path)
    print("Done!")

if __name__ == "__main__":
    main()