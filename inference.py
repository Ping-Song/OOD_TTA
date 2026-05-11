# inference.py

import argparse
import os
from typing import Dict, Callable

import pandas as pd
import json  # <- add this

from model import (
    T5SentimentModel,
    T5ToxicityModel,
    T5AGNewsModel,
    BertSentimentModel,
    BertToxicityModel,
    BertAGNewsModel,
    LlamaSentimentModel,
    LlamaToxicityModel,
    LlamaAGNewsModel,
    tta_on_dataframe,
)

# Short model tags for filenames
MODEL_SHORT_NAMES = {
    "bert-sentiment": "BERT",
    "bert-toxicity": "BERT",
    "bert-agnews": "BERT",
    "t5-sentiment": "T5",
    "t5-toxicity": "T5",
    "t5-agnews": "T5",
    "llama-sentiment": "LLAMA",
    "llama-toxicity": "LLAMA",
    "llama-agnews": "LLAMA", 
}

# Valid model names (also used for CLI choices)
MODEL_NAMES = list(MODEL_SHORT_NAMES.keys())


# -----------------------------
# Model registry
# -----------------------------
def build_model_registry(gpu_id: int) -> Dict[str, Callable[[], object]]:
    """
    Returns a mapping from model name (CLI string) to a zero-arg constructor.

    The gpu_id argument is passed to each model so the GPU can be selected
    from the command line (via --gpu-id).
    """
    return {
        "t5-sentiment": lambda: T5SentimentModel(device_id=gpu_id),
        "t5-toxicity": lambda: T5ToxicityModel(device_id=gpu_id),
        "t5-agnews": lambda: T5AGNewsModel(device_id=gpu_id),

        "bert-sentiment": lambda: BertSentimentModel(device_id=gpu_id),
        "bert-toxicity": lambda: BertToxicityModel(device_id=gpu_id),
        "bert-agnews": lambda: BertAGNewsModel(device_id=gpu_id),
        
        "llama-sentiment": lambda: LlamaSentimentModel(gpu_id=gpu_id),
        "llama-toxicity": lambda: LlamaToxicityModel(gpu_id=gpu_id),
        "llama-agnews": lambda: LlamaAGNewsModel(gpu_id=gpu_id),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference + TTA for BOSS models on a pickle DataFrame."
    )

    # REQUIRED
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_NAMES,
        help="Which model to use.",
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to .pkl file containing a pandas DataFrame with "
             "'original_text', 'label', and 'augmentation' columns.",
    )

    parser.add_argument(
        "--num-aug-used",
        type=int,
        required=False,
        help=(
            "How many augmentation predictions to include in the vote.\n"
            "0 = original only, 1 = original+1 aug, 2 = original+2 augs, etc."
        ),
    )

    # OPTIONAL
    parser.add_argument(
        "--tta-cols",
        nargs="+",
        default=["augmentation"],
        help="Augmentation columns to run TTA over (usually just 'augmentation').",
    )

    parser.add_argument(
        "--save-dir",
        default="predictions",
        help="Directory to save output .pkl with predictions.",
    )

    parser.add_argument(
        "--prefix",
        default="",
        help="Column name prefix for this model (e.g. 'bert_', 't5_', 'llama_').",
    )

    parser.add_argument(
        "--text-col",
        default="original_text",
        help="Name of the text column in the DataFrame.",
    )

    parser.add_argument(
        "--label-col",
        default="label",
        help="Name of the label column in the DataFrame.",
    )

    parser.add_argument(
        "--mode",
        choices=["both", "original", "tta"],
        default="both",
        help="Which accuracies to compute: 'both', 'original', or 'tta'.",
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU / device ID to use for model inference.",
    )

    args = parser.parse_args()

    # Enforce num-aug-used only when TTA is required
    if args.mode in ("both", "tta") and args.num_aug_used is None:
        parser.error("--num-aug-used is required when mode is 'both' or 'tta'.")

    # Load model (using the chosen GPU)
    registry = build_model_registry(args.gpu_id)
    model_fn = registry[args.model]
    model = model_fn()

    print(f"Using GPU ID: {args.gpu_id}")
    print(f"Loading DataFrame from {args.input!r} ...")
    df = pd.read_pickle(args.input)

    os.makedirs(args.save_dir, exist_ok=True)

    accuracies = []

    # -----------------------------
    # Run TTA on each augmentation column
    # -----------------------------
    for aug_col in args.tta_cols:
        if aug_col not in df.columns and args.mode != "original":
            print(f"[WARN] Column {aug_col!r} not found in DataFrame. Skipping.")
            continue

        compute_original = args.mode in ("both", "original")
        compute_tta = args.mode in ("both", "tta")

        df, orig_acc, tta_acc = tta_on_dataframe(
            df,
            aug_col_name=aug_col,
            predictor=model,
            text_col=args.text_col,
            label_col=args.label_col,
            prefix=args.prefix,
            max_augs_for_vote=args.num_aug_used,
            compute_original=compute_original,
            compute_tta=compute_tta,
        )
        accuracies.append((aug_col, orig_acc, tta_acc))

        # -------- Save accuracy results (moved from model.py) --------
        log_record = {
            "input_file": str(df.attrs.get("dataset_path", args.input)),
            "augmentation": aug_col,
            "model": args.model,
            "original_accuracy": float(orig_acc) if orig_acc is not None else None,
            "tta_accuracy": float(tta_acc) if tta_acc is not None else None,
            "max_augs_for_vote": args.num_aug_used,
            "gpu_id": args.gpu_id,
        }

        with open("tta_accuracy_log.jsonl", "a") as f:
            f.write(json.dumps(log_record) + "\n")

    # -----------------------------
    # Save output (pretty names)
    # -----------------------------
    base_name = os.path.basename(args.input).replace(".pkl", "")
    model_tag = MODEL_SHORT_NAMES.get(args.model, args.model.upper())
    k_tag = f"k{args.num_aug_used}"

    out_name = f"{base_name}_{model_tag}_{k_tag}_predicts.pkl"
    out_path = os.path.join(args.save_dir, out_name)

    df.to_pickle(out_path)
    print(f"\nSaved predictions to {out_path}")

    # -----------------------------
    # Print summary
    # -----------------------------
    print("\n=== ACCURACY SUMMARY ===")
    for aug_col, orig_acc, tta_acc in accuracies:
        orig_str = f"{orig_acc:.4f}" if orig_acc is not None else "N/A"
        tta_str = f"{tta_acc:.4f}" if tta_acc is not None else "N/A"
        print(f"{aug_col:16s} original={orig_str}  TTA({aug_col})={tta_str}")


if __name__ == "__main__":
    main()