#!/usr/bin/env python
"""
This script fine-tunes the pre-trained DistilBERT model (distilbert-base-uncased) on a given training and validation dataset for binary classification,
to check the performance on 10 different hyperparameter configurations sequentially, each with a unique seed.
For each configuration, the script records:
  - Training loss, accuracy, and AUC trajectories per epoch.
  - Validation loss, accuracy, and AUC trajectories per epoch.
  - The best validation loss (used for early stopping), the corresponding validation AUC, and the epoch at which the best performance was observed.
Early stopping is applied based solely on validation loss (with a patience of 4 epochs) to prevent overfitting.

Input: 



Output:
    - A JSON file containing the runhistory of all configurations (i.e runhistory of the training and validation metrics); \ 
       the winner configuration (lowest validation loss) is mentioned at the end
    - The winning configuration details are printed to the console.

Usage:
    python run_all_configs.py --base_seed 42 --train_file /path/to/train.json --val_file /path/to/val.json

"""

import argparse
import random
import numpy as np
import torch
import pandas as pd
import json
import os
import logging

from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TweetDataset(Dataset):
    """Custom Dataset for tokenized tweets and labels."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return item

def tokenize_function(texts, tokenizer, max_length=150):
    """Tokenizes a list/series of texts using the provided tokenizer."""
    return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")

def train_model(config, train_texts, train_labels, val_texts, val_labels, device, seed):
    """
    Fine-tunes DistilBERT with a given hyperparameter configuration and seed.
    Early stopping is applied based on validation loss with a patience of 4 epochs.
    
    Returns:
        model, tokenizer, and a dictionary of recorded metrics and trajectories.
    """
    set_seed(seed)  # Set seed for this specific configuration
    
    # Load the pre-trained model and tokenizer
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
    model.to(device)
    
    # Tokenize data
    train_encodings = tokenize_function(train_texts, tokenizer)
    val_encodings = tokenize_function(val_texts, tokenizer)
    
    # Create datasets and dataloaders
    train_dataset = TweetDataset(train_encodings, train_labels)
    val_dataset = TweetDataset(val_encodings, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    
    # Trajectories to record
    train_loss_traj, train_acc_traj, train_auc_traj = [], [], []
    val_loss_traj, val_acc_traj, val_auc_traj = [], [], []
    
    best_val_loss = float('inf')
    best_val_auc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    max_epochs = 9
    patience = 4
    
    for epoch in range(max_epochs):
        #------------------------
        # Training phase
        #------------------------
        model.train()
        total_train_loss = 0.0
        all_train_preds, all_train_labels, all_train_probs = [], [], []
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{max_epochs}", leave=False):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            all_train_probs.extend(probs.detach().cpu().numpy())
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_auc = roc_auc_score(all_train_labels, all_train_probs)
        train_loss_traj.append(avg_train_loss)
        train_acc_traj.append(train_acc)
        train_auc_traj.append(train_auc)
        
        #------------------------
        # Validation phase
        #------------------------
        model.eval()
        total_val_loss = 0.0
        all_val_preds, all_val_labels, all_val_probs = [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{max_epochs}", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                all_val_probs.extend(probs.detach().cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_auc = roc_auc_score(all_val_labels, all_val_probs)
        val_loss_traj.append(avg_val_loss)
        val_acc_traj.append(val_acc)
        val_auc_traj.append(val_auc)
        
        print(f"Epoch {epoch+1} (Seed {seed}): Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Train AUC={train_auc:.4f}; " +
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}, Val AUC={val_auc:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_auc = val_auc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (Seed {seed})")
                break
    
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
    
    results = {
        "hyperparameters": config,
        "seed": seed,
        "best_val_loss": best_val_loss,
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "train_loss_trajectory": train_loss_traj,
        "train_acc_trajectory": train_acc_traj,
        "train_auc_trajectory": train_auc_traj,
        "val_loss_trajectory": val_loss_traj,
        "val_acc_trajectory": val_acc_traj,
        "val_auc_trajectory": val_auc_traj,
        "final_train_loss": train_loss_traj[-1],
        "final_train_acc": train_acc_traj[-1],
        "final_train_auc": train_auc_traj[-1],
        "final_val_loss": val_loss_traj[-1],
        "final_val_acc": val_acc_traj[-1],
        "final_val_auc": val_auc_traj[-1]
    }
    
    return model, tokenizer, results

def main_loop():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility.")
    parser.add_argument("--train_file", type=str, default="/scratch/hpc-prf-eaitcm/rishi_thesis/train_data.json", help="Path to training data (JSON Lines).")
    parser.add_argument("--val_file", type=str, default="/scratch/hpc-prf-eaitcm/rishi_thesis/val_data.json", help="Path to validation data (JSON Lines).")
    args = parser.parse_args()
    
    base_seed = args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load training data
    train_df = pd.read_json(args.train_file, lines=True) # The content of this dataset is in jsonl format. Incase of json format content, then remove "lines=True"
    train_texts = train_df['tweets']
    train_labels = train_df['artificial']
    
    # Load validation data
    val_df = pd.read_json(args.val_file, lines=True)
    val_texts = val_df['tweets']
    val_labels = val_df['artificial']
    
    # Define 10 hyperparameter configurations
    configs = { 
        0: {"batch_size": 52, "learning_rate": 4.04657714e-05, "weight_decay": 0.256964152631},
        1: {"batch_size": 38, "learning_rate": 1.65368236e-05, "weight_decay": 0.0863231028617},
        2: {"batch_size": 39, "learning_rate": 3.71998137e-05, "weight_decay": 0.1250636563808},
        3: {"batch_size": 36, "learning_rate": 3.3161361e-05, "weight_decay": 0.2573639327469},
        4: {"batch_size": 35, "learning_rate": 3.2190498e-05, "weight_decay": 0.2571456003644},
        5: {"batch_size": 39, "learning_rate": 5.93668302e-05, "weight_decay": 0.1499933890691},
        6: {"batch_size": 51, "learning_rate": 6.57704526e-05, "weight_decay": 0.1425540162558},
        7: {"batch_size": 37, "learning_rate": 3.49886998e-05, "weight_decay": 0.1678043365548},
        8: {"batch_size": 48, "learning_rate": 7.34968617e-05, "weight_decay": 0.1714819321491},
        9: {"batch_size": 48, "learning_rate": 1.1423592e-05, "weight_decay": 0.0298972932962},
        10: {"batch_size": 37, "learning_rate": 5.07484213e-05, "weight_decay": 0.1359572500293},
    }
    
    all_results = {}
    
    # Loop over each configuration with a unique seed
    for idx in configs:
        config = configs[idx]
        seed = base_seed + idx  # Unique seed: 42, 43, ..., 51
        print(f"\n===== Running configuration {idx} with seed {seed} =====")
        
        model, tokenizer, results = train_model(config, train_texts, train_labels, val_texts, val_labels, device, seed)
        all_results[f"config_{idx}"] = results
        print(f"Configuration {idx} (Seed {seed}) results: Best Val Loss = {results['best_val_loss']:.4f}, Best Val AUC = {results['best_val_auc']:.4f}, Best Epoch = {results['best_epoch']}")
    
    # Identify the winning configuration
    winner = None
    best_loss = float('inf')
    best_auc = 0.0
    for key, res in all_results.items():
        if res["best_val_loss"] < best_loss or (res["best_val_loss"] == best_loss and res["best_val_auc"] > best_auc):
            best_loss = res["best_val_loss"]
            best_auc = res["best_val_auc"]
            winner = key
    
    # Print winner details
    winner_config_id = winner
    winner_results = all_results[winner_config_id]
    winner_config = winner_results["hyperparameters"]
    winner_seed = winner_results["seed"]
    print(f"\n===== Winner Configuration =====")
    print(f"Config ID: {winner_config_id}")
    print(f"Seed: {winner_seed}")
    print(f"Hyperparameters: Batch Size = {winner_config['batch_size']}, Learning Rate = {winner_config['learning_rate']:.8f}, Weight Decay = {winner_config['weight_decay']:.8f}")
    print(f"Best Validation Loss: {best_loss:.4f}")
    
    summary = {
        "winner_configuration": winner_config_id
    }
    all_results["summary"] = summary
    
    output_file = "results_of_all_configs.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to {output_file}")

if __name__ == "__main__":
    main_loop()