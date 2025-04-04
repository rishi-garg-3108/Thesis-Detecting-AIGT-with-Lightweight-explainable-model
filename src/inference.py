#!/usr/bin/env python

"""
This script loads fine-tuned transformer models from a directory,
evaluates them on various test datasets, and stores the evaluation metrics
(accuracy and AUC) into a JSON file.

Usage:
    python inference.py --mode_dir <path/to/model_dir> --test_dir <path/to/test_dir>
Arguments:
    --model_dir: Directory containing fine-tuned transformer models.
    --test_dir: Directory containing test datasets in JSON format.
Outputs:
    - A JSON file containing the evaluation metrics for each model on each test dataset.
    - A summary table printed to the console with average accuracy and AUC for each test dataset across all models.
Dependencies:
    - torch
    - transformers
    - sklearn
    - pandas
    - numpy
    - tqdm
    - tabulate
    - argparse
    - json
    - os
    - typing (List, Tuple, Dict)
This script is designed to be run in an environment where the required libraries are installed.
It is assumed that the models are compatible with the Hugging Face Transformers library.
The script is intended for evaluating transformer models on tweet classification tasks, specifically for detecting artificial tweets.
The script handles loading the models, tokenizing the input data, and calculating evaluation metrics.
Note: you can also run this script via a separate shell script incase your datasets are large. and compute intensive
Author: Rishi Garg
"""

import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm

class TweetDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: pd.Series):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return item

def load_test_dataset(file_path: str, tokenizer: AutoTokenizer) -> TweetDataset:
    """
    Load the test dataset from a JSON file and tokenize the tweets.
    Args:
        file_path (str): Path to the JSON file containing the test dataset.
        tokenizer (AutoTokenizer): Tokenizer for the transformer model.
    Returns:
        TweetDataset: A dataset object containing tokenized tweets and labels.
    """
    
    df = pd.read_json(file_path, lines=True)
    df = df[['tweets', 'artificial']]

    encodings = tokenizer(
        df['tweets'].tolist(),
        padding=True,
        truncation=True,
        max_length=150,
        return_tensors="pt"
    )

    dataset = TweetDataset(encodings, df['artificial'])
    return dataset

def evaluate_model(model: AutoModelForSequenceClassification, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate the model on the given dataloader and return accuracy and AUC.
    Args:
        model (AutoModelForSequenceClassification): The transformer model to evaluate.
        dataloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the model on (CPU or GPU).
    Returns:
        Tuple[float, float]: Accuracy and AUC of the model on the test dataset.
    """ 
    
    model.eval()
    all_preds: List[int] = []
    all_probs: List[float] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return acc, auc

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the directory containing fine-tuned models")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to the directory containing test dataset JSON files")
    args = parser.parse_args()

    model_base_dir: str = args.model_dir
    test_dir: str = args.test_dir
    batch_size: int = 32
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dirs: List[str] = [
        os.path.join(model_base_dir, d)
        for d in os.listdir(model_base_dir)
        if os.path.isdir(os.path.join(model_base_dir, d))
    ]

    test_files: List[str] = sorted([
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.endswith(".json")
    ])

    if not test_files:
        raise ValueError("No test datasets found in the given test directory.")

    results: Dict[str, Dict] = {}
    dataset_metrics: Dict[str, Dict[str, List[float]]] = {
        os.path.splitext(os.path.basename(f))[0]: {'acc': [], 'auc': []} for f in test_files
    }

    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            print(f"Warning: Model directory {model_dir} does not exist, skipping...")
            continue

        print(f"\n===== Evaluating Model: {model_dir} =====")

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
        model.to(device)

        model_results: Dict[str, Dict[str, float]] = {}

        for test_file in test_files:
            test_name = os.path.splitext(os.path.basename(test_file))[0]
            print(f"  Evaluating on {test_name}...")

            test_dataset = load_test_dataset(test_file, tokenizer)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            acc, auc = evaluate_model(model, test_dataloader, device)
            model_results[test_name] = {
                "accuracy": acc,
                "auc": auc,
                "num_samples": len(test_dataset)
            }
            dataset_metrics[test_name]['acc'].append(acc)
            dataset_metrics[test_name]['auc'].append(auc)
            print(f"    {test_name}: ACC={acc:.4f}, AUC={auc:.4f}, Samples={len(test_dataset)}")

        results[model_dir] = model_results
        print("----------")

    averages: Dict[str, Dict[str, float]] = {}
    table_data: List[List[str]] = []
    print("\n===== Average Performance Across Models =====")
    for test_name, metrics in dataset_metrics.items():
        avg_acc = np.mean(metrics['acc'])
        std_acc = np.std(metrics['acc'])
        avg_auc = np.mean(metrics['auc'])
        std_auc = np.std(metrics['auc'])
        averages[test_name] = {
            "avg_accuracy": float(avg_acc),
            "std_accuracy": float(std_acc),
            "avg_auc": float(avg_auc),
            "std_auc": float(std_auc),
            "num_models": len(metrics['acc'])
        }
        table_data.append([test_name, f"{avg_acc:.4f} ± {std_acc:.4f}", f"{avg_auc:.4f} ± {std_auc:.4f}"])

    results["averages"] = averages

    output_file = "inference_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    print("\n===== Summary Table =====")
    print(tabulate(table_data, headers=["Test Dataset", "Avg Accuracy ± Std", "Avg AUC ± Std"], tablefmt="grid"))

if __name__ == "__main__":
    main()
