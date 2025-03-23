"""This script calculates the perplexity of tweets from a CSV or JSON file.
It splits the tweets into two classes: original (label=0) and fake (label=1).
It uses the HuggingFace evaluate library to compute the perplexity scores
for each class and prints the results.
Usage:
    python run_perplexity.py --input <path_to_input_file> --limit <max_tweets_per_class> --model_id
    <model_id>
"""

"""Caution: If the script gives error, try running the ipynb notebook
'run_perplexity.ipynb' instead.
 """



# -*- coding: utf-8 -*-


import argparse
import pandas as pd
import json
import evaluate
import os
import numpy as np

def load_and_split_tweets(input_path, limit=None):
    """
    Loads tweets from a CSV or JSON file, splits them into original and fake tweets
    based on either the 'label' or 'artificial' column/key.
    The text column can be either 'tweet' or 'tweets'.

    Args:
        input_path (str): Path to input file (.csv or .json)
        limit (int): Max number of tweets to process from each class

    Returns:
        original_tweets (list of str)
        fake_tweets (list of str)
    """

    # 1. Determine file extension and read accordingly
    ext = os.path.splitext(input_path)[-1].lower()

    if ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError("Only .csv and .json files are supported.")

    # 2. Detect possible text column: either 'tweet' or 'tweets'
    if "tweet" in df.columns:
        text_col = "tweet"
    elif "tweets" in df.columns:
        text_col = "tweets"
    else:
        raise ValueError("No text column found. Must be 'tweet' or 'tweets'.")

    # 3. Detect possible label column: either 'label' or 'artificial'
    if "label" in df.columns:
        label_col = "label"
    elif "artificial" in df.columns:
        label_col = "artificial"
    else:
        raise ValueError("No label column found. Must be 'label' or 'artificial'.")

    # 4. Drop rows with missing text or label
    df = df.dropna(subset=[text_col, label_col])

    # 5. Filter tweets based on label (0 or 1)
    original = df[df[label_col] == 0][text_col].astype(str).tolist()
    fake = df[df[label_col] == 1][text_col].astype(str).tolist()

    # 6. Apply limit if specified
    if limit:
        original = original[:limit]
        fake = fake[:limit]

    return original, fake


def compute_perplexity(texts, model_id="gpt2"):
    """
    Computes perplexity scores using HuggingFace evaluate module.

    Args:
        texts (list of str): List of input texts
        model_id (str): Pretrained model to use (default: gpt2)

    Returns:
        dict: {
            'perplexities': list of perplexity scores,
            'mean': average perplexity
        }
    """
    metric = evaluate.load("perplexity", module_type="metric")
    results = metric.compute(model_id=model_id, predictions=texts)

    return {
        "perplexities": results["perplexities"],
        "mean": np.mean(results["perplexities"])
    }


def main():
    parser = argparse.ArgumentParser(description="Compute perplexity scores for human vs machine-generated tweets.")
    parser.add_argument("--input", required=True, help="Path to input file (.csv or .json)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of tweets per class to process")
    parser.add_argument("--model_id", default="gpt2", help="HuggingFace model ID to use (default: gpt2)")

    args = parser.parse_args()

    try:
        # Load and split tweets
        original_tweets, fake_tweets = load_and_split_tweets(args.input, limit=args.limit)

        print(f"Original tweets loaded: {len(original_tweets)}")
        print(f"Fake tweets loaded: {len(fake_tweets)}")

        print("\nComputing perplexity for original (label=0) tweets...")
        orig_result = compute_perplexity(original_tweets, model_id=args.model_id)

        print(" Computing perplexity for fake (label=1) tweets...")
        fake_result = compute_perplexity(fake_tweets, model_id=args.model_id)

        # Display results
        print("\n📊 Results:")
        print(f"Original Tweets (label=0): Mean Perplexity = {orig_result['mean']:.2f}")
        print(f"Fake Tweets (label=1): Mean Perplexity = {fake_result['mean']:.2f}")

    except Exception as e:
        print(f" Error: {e}")


if __name__ == "__main__":
    main()
