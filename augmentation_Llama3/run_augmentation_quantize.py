
import torch
import pandas as pd
import numpy as np
import argparse
import os
import re
import openai

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Load API Key securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ==========================================
# 1. Define Your Prompt Library
# ==========================================
def get_prompt(prompt_type, example_sentences, input_sentence):
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
    cleaned = re.sub(r'^[\s"\']*paraphrased text:\s*', '', cleaned, flags=re.IGNORECASE)

    stop_pattern = r"(###|Instruction|Input Sentence|Rewrite|17\.)"
    match = re.search(stop_pattern, cleaned, flags=re.IGNORECASE)
    if match:
        cleaned = cleaned[:match.start()]

    cleaned = cleaned.strip().strip('"').strip("'").strip()
    return cleaned


def query_model(model, tokenizer, prompt, model_type, temperature=0.7, max_tokens=256):
    if model_type == "gpt-4o":
        completion = model.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at transforming sentences to other domains and shift semantic meaning to something in that domain."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content.strip()

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
        try:
            eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None and eot_id != tokenizer.unk_token_id:
                terminators.append(eot_id)
        except Exception:
            pass

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
    rng = np.random.RandomState(random_state)

    for label in unique_labels:
        df_label = df[df[label_col] == label]
        count = min(base_count, len(df_label))
        sampled_list.append(df_label.sample(n=count, random_state=random_state))

    extra_labels = rng.choice(unique_labels, size=remainder, replace=False) if remainder > 0 else []
    already_sampled = pd.concat(sampled_list).index if sampled_list else pd.Index([])

    for label in extra_labels:
        df_label = df[(df[label_col] == label) & (~df.index.isin(already_sampled))]
        if not df_label.empty:
            sampled_list.append(df_label.sample(n=1, random_state=random_state))

    sampled_df = pd.concat(sampled_list).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return sampled_df["original_text"].tolist()


def augmentation(df_id, df_ood, model, tokenizer, prompt_type, model_type, num_augs=4, examples_per_prompt=16):
    df_ood["augmentation"] = df_ood["augmentation"].astype(object)

    for index, row in df_ood.iterrows():
        tem_aug = []
        for i in range(num_augs):
            example_sentences = balanced_sample(df_id, n_samples=examples_per_prompt, label_col="label")
            input_sentence = row["original_text"]

            prompt = get_prompt(prompt_type, example_sentences, input_sentence)
            raw_aug = query_model(model, tokenizer, prompt, model_type)
            tem_aug.append(clean_output(raw_aug))

        df_ood.at[index, "augmentation"] = tem_aug

        if index % 50 == 0:
            print(f"Processed {index} rows")

    return df_ood


# ==========================================
# 3. Model Loading with Quantization
# ==========================================
def load_local_model(model_id, augmentation_model, quantization):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantization == "none":
        dtype = torch.float16 if augmentation_model == "qwen" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto"
        )

    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map="auto"
        )

    elif quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if augmentation_model == "llama3" else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map="auto"
        )

    else:
        raise ValueError(f"Unknown quantization mode: {quantization}")

    model.eval()
    return model, tokenizer


# ==========================================
# 4. Main Execution Block
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Run LLM Data Augmentation")
    parser.add_argument("--id_data", type=str, required=True, help="Path to In-Domain dataset")
    parser.add_argument("--ood_data", type=str, required=True, help="Path to Out-Of-Domain dataset")
    parser.add_argument("--prompt_type", type=str, choices=["sentiment", "toxicity", "agnews", "icr"], required=True, help="Prompt template to use")
    parser.add_argument("--augmentation_model", type=str, choices=["llama3", "qwen", "gpt-4o"], required=True, help="Model to use: 'llama3', 'qwen' or 'gpt-4o'")
    parser.add_argument("--quantization", type=str, choices=["none", "8bit", "4bit"], default="none",
                        help="Quantization mode for local HF models")
    parser.add_argument("--num_augs", type=int, default=4, help="Number of augmentations per input")
    parser.add_argument("--examples_per_prompt", type=int, default=16, help="Number of ID exemplars per prompt")
    parser.add_argument("--output_file", type=str, default=None, help="Optional: Force a specific output path")

    args = parser.parse_args()
    np.random.seed(42)

    if args.output_file is None:
        dataset_name = os.path.splitext(os.path.basename(args.ood_data))[0]
        output_dir = f"/home/pingsong/OOD_TTA/augmentation_{args.augmentation_model}"
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(
            output_dir,
            f"{dataset_name}_{args.prompt_type}_{args.quantization}.pkl"
        )
    else:
        out_path = args.output_file

    print("Loading data...")
    df_id = pd.read_pickle(args.id_data)
    df_ood = pd.read_pickle(args.ood_data)
    df_ood.insert(1, "augmentation", pd.Series([[] for _ in range(len(df_ood))], dtype=object))

    if args.augmentation_model == "gpt-4o":
        print("Using OpenAI client for GPT-4o...")
        model = openai.OpenAI()
        tokenizer = None

    else:
        if args.augmentation_model == "llama3":
            model_id = "meta-llama/Llama-3.1-8B-Instruct"
        elif args.augmentation_model == "qwen":
            model_id = "Qwen/Qwen2.5-7B-Instruct"
        else:
            raise ValueError(f"Unknown local model: {args.augmentation_model}")

        print(f"Loading {model_id} with quantization={args.quantization}...")
        try:
            model, tokenizer = load_local_model(
                model_id=model_id,
                augmentation_model=args.augmentation_model,
                quantization=args.quantization
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model with quantization={args.quantization}. "
                f"If using 4bit/8bit, make sure bitsandbytes is installed and supported on your system.\n"
                f"Original error: {e}"
            )

    print(
        f"Starting augmentation with prompt='{args.prompt_type}', "
        f"model={args.augmentation_model}, quantization={args.quantization}..."
    )

    df_ood = augmentation(
        df_id=df_id,
        df_ood=df_ood,
        model=model,
        tokenizer=tokenizer,
        prompt_type=args.prompt_type,
        model_type=args.augmentation_model,
        num_augs=args.num_augs,
        examples_per_prompt=args.examples_per_prompt
    )

    df_ood["augmentation"] = df_ood["augmentation"].apply(lambda lst: [s.strip() for s in lst])

    print(f"Saving to {out_path}...")
    df_ood.to_pickle(out_path)
    print("Done!")


if __name__ == "__main__":
    main()