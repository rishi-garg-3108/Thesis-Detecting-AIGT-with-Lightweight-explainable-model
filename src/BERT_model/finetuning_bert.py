#!/usr/bin/env python
"""
This script fine-tunes the DistilBERT model for sequence classification on a custom dataset.
It accepts a dataset file as input and automatically determines whether the file is in JSON Lines 
(.jsonl) format or standard JSON (.json) format based on the file extension.
The script then splits the data into training and validation sets, tokenizes the text,
creates PyTorch Datasets and DataLoaders, and trains the model using early stopping.
The best model checkpoint and training results (validation loss and AUC) are saved to disk.

Usage examples:
    python single_run.py --run_index 0 --seed 42 --data_path /path/to/dataset.jsonl
    python single_run.py --run_index 0 --seed 42 --data_path /path/to/dataset.json
    
    Or
    
    chmod +x single_run.py
    ./single_run.py --run_index 0 --seed 42 --data_path /path/to/dataset.jsonl/json
"""

import argparse
import random
import numpy as np
import torch
import pandas as pd
import json
import os

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): The random seed to use for Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TweetDataset(Dataset):
    """
    PyTorch Dataset for loading tweet data and labels.

    Attributes:
        encodings (dict): Tokenized representations of the tweets.
        labels (pd.Series): Corresponding labels for each tweet.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        # Returns the number of samples in the dataset.
        return len(self.labels)

    def __getitem__(self, idx):
        # Retrieves a single sample (tokenized tweet and its label).
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return item


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_index", type=int, default=0, help="Index of the run (0-9).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset file (JSON or JSON Lines).")
    args = parser.parse_args()

    run_index = args.run_index
    run_seed = args.seed
    data_path = args.data_path

    # --- 1) Set seed for reproducibility ---
    set_seed(run_seed)
    print(f"\n===== Starting Run {run_index} with Seed {run_seed} =====")

    # --- 2) Load Data ---
    # Automatically check the file extension to determine the format.
    _, file_extension = os.path.splitext(data_path)
    file_extension = file_extension.lower()
    if file_extension == ".jsonl":
        print("Detected JSON Lines file format.")
        df = pd.read_json(data_path, lines=True)
    else:
        print("Detected standard JSON file format.")
        df = pd.read_json(data_path)
    
    # Select only the necessary columns.
    df = df[['tweets', 'artificial']]

    # --- 3) Train/Validation Split ---
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['tweets'], df['artificial'],
        test_size=0.1,              # 10% of data for validation.
        random_state=run_seed       # Ensure reproducibility of the split.
    )
    
    # Log the number of samples.
    print(f"Number of training samples: {len(train_texts)}")
    print(f"Number of validation samples: {len(val_texts)}")

    # --- 4) Tokenization ---
    # Load the tokenizer for DistilBERT.
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(texts):
        """
        Tokenizes a list/series of texts.

        Args:
            texts (list or pd.Series): The texts to tokenize.

        Returns:
            dict: A dictionary of tokenized tensors.
        """
        return tokenizer(
            texts.tolist(),           # Convert texts to list.
            padding=True,             # Pad sequences to the same length.
            truncation=True,          # Truncate sequences to max_length.
            max_length=150,           # Maximum token length.
            return_tensors="pt"       # Return PyTorch tensors.
        )

    # Tokenize the training and validation texts.
    train_encodings = tokenize_function(train_texts)
    val_encodings   = tokenize_function(val_texts)

    # --- 5) Create Datasets & DataLoaders ---
    # Create custom Dataset objects.
    train_dataset = TweetDataset(train_encodings, train_labels)
    val_dataset   = TweetDataset(val_encodings,   val_labels)

    # Create DataLoaders for batching.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32)
    
    # Log number of batches in each DataLoader.
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # --- 6) Initialize the Model ---
    # Load DistilBERT for sequence classification (binary classification).
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- 7) Define Optimizer and Loss Function ---
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # --- 8) Training Loop with Early Stopping ---
    epochs = 8                   # Total number of epochs.
    early_stop_patience = 2      # Stop if no improvement for 2 consecutive epochs.

    best_val_loss = float('inf') # Initialize best validation loss.
    best_model_dir = f"best_model_run_{run_index}"  # Directory to save the best model.
    epochs_no_improve = 0        # Counter for early stopping.
    final_val_auc = 0.0          # To store final epoch's AUC.

    for epoch in range(epochs):
        # ---- Training Phase ----
        model.train()  # Set model to training mode.
        total_train_loss = 0
        train_preds, train_labels_list = [], []

        # Use tqdm for a progress bar over the training batches.
        pbar = tqdm(train_loader, desc=f"Run {run_index}, Epoch {epoch+1}/{epochs}", leave=False)
        for i, batch in enumerate(pbar):
            # Move batch to device.
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()  # Clear gradients.
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits
            loss    = criterion(logits, labels)

            # Backpropagation.
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Compute predictions and store them for training accuracy.
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())

            # Update progress bar; use (i+1) to avoid division by zero.
            pbar.set_postfix({"train_loss": total_train_loss / (i + 1)})

        # Compute average training loss and accuracy.
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels_list, train_preds)
        print(f"Run {run_index}, Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}")

        # ---- Validation Phase ----
        model.eval()  # Set model to evaluation mode.
        total_val_loss = 0
        val_preds = []
        val_true  = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits  = outputs.logits

                # Compute validation loss for the batch.
                val_loss = criterion(logits, labels)
                total_val_loss += val_loss.item()

                # Compute probability of positive class for AUC.
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                val_preds.extend(probs)
                val_true.extend(labels.cpu().numpy())

        # Compute average validation loss and AUC.
        avg_val_loss = total_val_loss / len(val_loader)
        val_auc = roc_auc_score(val_true, val_preds)
        print(f"Run {run_index}, Epoch {epoch+1}: Val Loss={avg_val_loss:.4f}, Val AUC={val_auc:.4f}")
        final_val_auc = val_auc

        # ---- Early Stopping Check ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0  # Reset the counter.
            # Save the best model and tokenizer.
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print(f"  [BEST] => Loss={best_val_loss:.4f}, AUC={val_auc:.4f} (saved)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"  [EARLY STOP] at epoch={epoch+1}")
                break

    # --- 9) Save Final Results ---
    results_dict = {
        "run_index": run_index,
        "seed": run_seed,
        "best_val_loss": best_val_loss,
        "final_val_auc": final_val_auc,
        "best_model_dir": best_model_dir
    }
    with open(f"results_run_{run_index}.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n===== Run {run_index} complete. Best val loss={best_val_loss:.4f}, AUC={final_val_auc:.4f} =====")


if __name__ == "__main__":
    main()
