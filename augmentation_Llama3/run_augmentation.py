import torch
import pandas as pd
import numpy as np
import argparse
import os
import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import openai


# Load API Key securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

Return the text in the format: ```Paraphrased Text```

### Transformed Text ###
Paraphrased Text:
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
    if text is None:
        return ""

    cleaned = str(text).strip()

    # Remove common leading wrappers like:
    # Paraphrased Text:
    # "Paraphrased Text:
    # 'Paraphrased Text:
    cleaned = re.sub(
        r'^[\s"\']*paraphrased text:\s*',
        '',
        cleaned,
        flags=re.IGNORECASE
    )

    # Remove trailing prompt leakage
    stop_pattern = r"(###|Instruction|Input Sentence|Rewrite|17\.)"
    match = re.search(stop_pattern, cleaned, flags=re.IGNORECASE)
    if match:
        cleaned = cleaned[:match.start()]

    # Strip surrounding quotes again after cleanup
    cleaned = cleaned.strip().strip('"').strip("'").strip()

    return cleaned

def query_model(model, tokenizer, prompt, model_type, temperature=0.7, max_tokens=256):
    # Route for GPT-4o
    if model_type == "gpt-4o":
        # 'model' is passed as the instantiated OpenAI client
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at transforming sentences to other domains and shift semantic meaning to something in that domain."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content.strip()

    # Route for local Hugging Face Models
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant that strictly follows instructions to transform text."},
        {"role": "user", "content": prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    input_length = inputs.input_ids.shape[-1]
    
    safe_max_new = max(1, min(max_tokens, 4096 - input_length))

    # Handle different stopping tokens for Llama vs Qwen
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

    for index, row in df_ood.iterrows():
        tem_aug = []
        for i in range(4):
            example_sentences = balanced_sample(df_id, n_samples=16, label_col='label') 
            input_sentence = row["original_text"]
            
            prompt = get_prompt(prompt_type, example_sentences, input_sentence)
            
            # Pass model_type to handle specific generation rules
            raw_aug = query_model(model, tokenizer, prompt, model_type)
            tem_aug.append(clean_output(raw_aug))

        df_ood.at[index, 'augmentation'] = tem_aug 

        if index % 50 == 0:
            print(f"Processed {index} rows")

    return df_ood

# ==========================================
# 3. Latency Benchmark
# ==========================================
def benchmark_latency(
    df_id,
    df_ood,
    model,
    tokenizer,
    prompt_type,
    model_type,
    num_augs=4,
    benchmark_rows=50,
    warmup_rows=5,
    examples_per_prompt=16,
    max_tokens=256,
    temperature=0.7,
    random_state=42,
):
    """
    Measures end-to-end wall-clock latency for augmentation only:
    prompt construction + generation + cleaning, across num_augs rewrites per sample.
    """
    assert benchmark_rows > 0
    subset = df_ood.head(benchmark_rows).copy()

    # Warmup to stabilize runtime
    if warmup_rows > 0:
        warmup_subset = subset.head(min(warmup_rows, len(subset)))
        print(f"Running {len(warmup_subset)} warmup rows...")
        for _, row in warmup_subset.iterrows():
            for _ in range(num_augs):
                example_sentences = balanced_sample(
                    df_id, n_samples=examples_per_prompt, label_col="label", random_state=random_state
                )
                prompt = get_prompt(prompt_type, example_sentences, row["original_text"])
                raw = query_model(
                    model, tokenizer, prompt, model_type,
                    temperature=temperature, max_tokens=max_tokens
                )
                _ = clean_output(raw)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    print(f"Benchmarking {len(subset)} rows with prompt_type={prompt_type}, num_augs={num_augs}...")

    total_start = time.perf_counter()

    per_sample_times = []
    total_generated_chars = 0

    for idx, row in subset.iterrows():
        sample_start = time.perf_counter()

        for aug_i in range(num_augs):
            example_sentences = balanced_sample(
                df_id,
                n_samples=examples_per_prompt,
                label_col="label",
                random_state=random_state + idx + aug_i
            )
            prompt = get_prompt(prompt_type, example_sentences, row["original_text"])
            raw = query_model(
                model,
                tokenizer,
                prompt,
                model_type,
                temperature=temperature,
                max_tokens=max_tokens
            )
            cleaned = clean_output(raw)
            total_generated_chars += len(cleaned)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        sample_elapsed = time.perf_counter() - sample_start
        per_sample_times.append(sample_elapsed)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_elapsed = time.perf_counter() - total_start

    avg_latency_per_sample = float(np.mean(per_sample_times))
    std_latency_per_sample = float(np.std(per_sample_times))
    avg_latency_per_aug = avg_latency_per_sample / num_augs
    avg_generated_chars_per_sample = total_generated_chars / max(1, len(subset))

    results = {
        "prompt_type": prompt_type,
        "model_type": model_type,
        "num_rows": len(subset),
        "num_augs": num_augs,
        "examples_per_prompt": examples_per_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "total_elapsed_sec": round(total_elapsed, 4),
        "avg_latency_per_sample_sec": round(avg_latency_per_sample, 4),
        "std_latency_per_sample_sec": round(std_latency_per_sample, 4),
        "avg_latency_per_aug_sec": round(avg_latency_per_aug, 4),
        "avg_generated_chars_per_sample": round(avg_generated_chars_per_sample, 2),
    }
    return results


def save_latency_result(result, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save JSON
    json_path = out_path if out_path.endswith(".json") else out_path + ".json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # Also append/save CSV-friendly version
    csv_path = json_path.replace(".json", ".csv")
    pd.DataFrame([result]).to_csv(csv_path, index=False)

    print(f"Saved latency JSON to {json_path}")
    print(f"Saved latency CSV to {csv_path}")

# ==========================================
# 4. Main Execution Block
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Run LLM Data Augmentation")
    parser.add_argument("--id_data", type=str, required=True, help="Path to In-Domain dataset")
    parser.add_argument("--ood_data", type=str, required=True, help="Path to Out-Of-Domain dataset")
    parser.add_argument("--prompt_type", type=str, choices=["sentiment", "toxicity", "agnews", "icr"], required=True, help="Prompt template to use")
    parser.add_argument("--augmentation_model", type=str, choices=["llama3", "qwen", "gpt-4o"], required=True, help="Model to use: 'llama3', 'qwen' or 'gpt-4o'")
    parser.add_argument("--output_file", type=str, default=None, help="Optional: Force a specific output path")

    # Latency benchmark arguments
    parser.add_argument("--benchmark_latency", action="store_true", help="Run latency benchmark instead of augmentation")
    parser.add_argument("--num_augs", type=int, default=4, help="Number of augmentations per input")
    parser.add_argument("--benchmark_rows", type=int, default=50, help="Number of rows to benchmark")
    parser.add_argument("--warmup_rows", type=int, default=5, help="Number of warmup rows before timing")
    parser.add_argument("--examples_per_prompt", type=int, default=16, help="Number of ID exemplars per prompt")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    
    args = parser.parse_args()
    np.random.seed(42)

    # Automatically construct output filename if not provided
    if args.output_file is None:
        dataset_name = os.path.splitext(os.path.basename(args.ood_data))[0]
        output_dir = f"/home/pingsong/OOD_TTA/augmentation_{args.augmentation_model}"
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{dataset_name}_{args.prompt_type}.pkl")
    else:
        out_path = args.output_file

    print(f"Loading data...")
    df_id = pd.read_pickle(args.id_data)
    df_ood = pd.read_pickle(args.ood_data)
    df_ood.insert(1, "augmentation", pd.Series([[] for _ in range(len(df_ood))], dtype=object))

    # --- THE FIX: Initialize these as None so they always exist ---
    model = None
    tokenizer = None
    # --------------------------------------------------------------

    # Strict Model Setup Logic
    if args.augmentation_model == "gpt-4o":
        print("Using global OpenAI client for GPT-4o...")
        
    else:
        # Everything here is STRICTLY for Hugging Face models
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

        if args.benchmark_latency:
            dataset_name = os.path.splitext(os.path.basename(args.ood_data))[0]
            if args.output_file is None:
                output_dir = f"/home/pingsong/OOD_TTA/latency_{args.augmentation_model}"
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, f"{dataset_name}_{args.prompt_type}_latency")
            else:
                out_path = args.output_file

            results = benchmark_latency(
                df_id=df_id,
                df_ood=df_ood,
                model=model,
                tokenizer=tokenizer,
                prompt_type=args.prompt_type,
                model_type=args.augmentation_model,
                num_augs=args.num_augs,
                benchmark_rows=args.benchmark_rows,
                warmup_rows=args.warmup_rows,
                examples_per_prompt=args.examples_per_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                random_state=42,
            )

            print("\nLatency Benchmark Results")
            print("-" * 40)
            for k, v in results.items():
                print(f"{k}: {v}")

            save_latency_result(results, out_path)
            return

        # Normal augmentation path
        if args.output_file is None:
            dataset_name = os.path.splitext(os.path.basename(args.ood_data))[0]
            output_dir = f"/home/pingsong/OOD_TTA/augmentation_{args.augmentation_model}"
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{dataset_name}_{args.prompt_type}.pkl")
        else:
            out_path = args.output_file


    print(f"Starting augmentation with '{args.prompt_type}' prompt using {args.augmentation_model}...")
    
    # Now this line will work perfectly because model/tokenizer are at least defined as `None` for gpt-4o
    df_ood = augmentation(df_id, df_ood, model, tokenizer, args.prompt_type, args.augmentation_model)

    df_ood["augmentation"] = df_ood["augmentation"].apply(lambda lst: [s.strip() for s in lst])
    
    print(f"Saving to {out_path}...")
    df_ood.to_pickle(out_path)
    print("Done!")

if __name__ == "__main__":
    main()