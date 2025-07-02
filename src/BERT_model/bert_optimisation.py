#!/usr/bin/env python
"""
1. This script uses SMAC to optimize hyperparameters for fine-tuning the pre-trained DistilBERT model
(distilbert-base-uncased) on a dataset. 
2. It runs independent SMAC runs with different seeds.
3. The target function returns the validation loss.
The best configuration and results are saved for each run.

NOTE: This script is designed to run in a SLURM job array environment.
It assumes the dataset is in JSON Lines format, where each line is a JSON object with "tweets" and "artificial" keys.
The "tweets" key contains the text of the tweet, and the "artificial" key contains the label (0 or 1).
The script uses PyTorch for model training and evaluation, and the Hugging Face Transformers library for the DistilBERT model.
It also uses the SMAC library for hyperparameter optimization.


Make sure you have the required libraries installed!
Please alter the paths in the script according to your needs, especially the dataset path and smac output path!
"""

import os
import json
import sys
import logging
import time
import numpy as np
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import SGD
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from smac import HyperparameterOptimizationFacade, Scenario
from tqdm.auto import tqdm

# Setup logging to print messages to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


# Seed Setup
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# Dataset class for tokenized tweets
class TweetDataset(Dataset):
    """Custom Dataset for tokenized tweets and labels."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ==========================================================================================================
class BertClassifier:
    @property
    def configspace(self) -> ConfigurationSpace:
        """
        Define the hyperparameter configuration space for optimizing DistilBERT.
        The space includes learning_rate, weight_decay, batch_size, and optimizer.
        """
        cs = ConfigurationSpace(seed=0)
        lr = Float("learning_rate", (1e-7, 1e-4), default=2e-5, log=True)
        weight_decay = Float("weight_decay", (0.0, 0.3), default=0.01)
        batch_size = Integer("batch_size", (32, 64), default=32)
        optimizer = Categorical("optimizer", ["adamw", "sgd"], default="adamw")
        cs.add_hyperparameters([lr, weight_decay, batch_size, optimizer])
        return cs

    def train(self, config, seed=42) -> float:
        """
        Fine-tunes DistilBERT using a custom PyTorch training loop.
        The model is fine-tuned on a dataset that is split into 90% training and 10% validation.
        Early stopping is applied on the validation loss with a patience of 3 epochs.
        
        The function returns the best validation loss, which SMAC will minimize.
        """
        # Extract hyperparameters from SMAC configuration
        config_dict = config.get_dictionary()
        lr = config_dict["learning_rate"]
        weight_decay = config_dict["weight_decay"]
        batch_size = config_dict["batch_size"]
        optimizer_choice = config_dict["optimizer"]
        
        logging.info(f"Training with config: {config_dict}")

        # Load dataset (assumed to be in JSON Lines format)
        dataset_path = '/scratch/hpc-prf-eaitcm/rishi_thesis/train_random_gpt35_fewshot_withoutLink_without_emojis.json'
        with open(dataset_path, "r") as f:
            data = [json.loads(line) for line in f]
        
        # Use a fixed random state for the validation split (as per supervisor's suggestion)
        train_data, val_data = train_test_split(
            data, test_size=0.5, random_state=seed, stratify=[d["artificial"] for d in data])
        
        train_texts = [d["tweets"] for d in train_data]
        train_labels = [d["artificial"] for d in train_data]
        val_texts = [d["tweets"] for d in val_data]
        val_labels = [d["artificial"] for d in val_data]
        
        # Load the pre-trained DistilBERT model and tokenizer.
        model_checkpoint = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Tokenize data. Use simple tokenization (padding/truncation).
        max_length = 150
        def tokenize_fn(texts):
            return tokenizer(texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        
        train_encodings = tokenize_fn(train_texts)
        val_encodings = tokenize_fn(val_texts)
        
        # Create datasets and dataloaders
        train_dataset = TweetDataset(train_encodings, train_labels)
        val_dataset = TweetDataset(val_encodings, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        if optimizer_choice.lower() == "sgd":
            custom_optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            custom_optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        #================================================== TRAINING LOOP ==========================================================
        # Training loop parameters
        max_epochs = 3  # Number of epochs to train; for optimisation, it was set to 3
        best_val_loss = float('inf')
        best_val_auc = 0.0
        best_epoch = 0
        best_model_state = None

        for epoch in range(max_epochs):
            # ---------------------------
            # TRAINING PHASE
            # ---------------------------
            model.train()
            total_train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{max_epochs}", leave=False):
                custom_optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
                
                loss.backward()
                custom_optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # ---------------------------
            # VALIDATION PHASE
            # ---------------------------
            model.eval()
            total_val_loss = 0.0
            all_val_probs = []
            all_val_labels = []
    
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{max_epochs}", leave=False):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                    total_val_loss += loss.item()

                    probs = torch.softmax(logits, dim=1)[:, 1]
                    all_val_probs.extend(probs.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            val_auc = roc_auc_score(all_val_labels, all_val_probs)

            logging.info(
                f"Epoch {epoch+1}/{max_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val AUC: {val_auc:.4f}"
            )

            # ---------------------------
            # TRACK BEST MODEL
            # ---------------------------
            # No early stopping: we always let all epochs run,
            # but we keep track of whichever epoch yields the best val loss.
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_auc = val_auc
                best_epoch = epoch + 1
                best_model_state = model.state_dict()

        # End of all epochs. Now reload the best model state:
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logging.info(
                f"Best model reloaded from epoch {best_epoch} "
                f"with Val Loss={best_val_loss:.4f}, AUC={best_val_auc:.4f}"
            )

        # The function can return best_val_loss (or whichever metric you want)
        return best_val_loss

# ==========================================================================================================

if __name__ == "__main__":
    start_time = time.time()
    logging.info("SMAC optimization started.")
    
    # Create an instance of the classifier
    classifier = BertClassifier()
    
    # Get run ID from SLURM_ARRAY_TASK_ID (or default to 0)
    run_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    seed = 42 + run_id  # Unique seed per run
    logging.info(f"Optimizing for run {run_id} with seed {seed}")
    
    # SMAC Scenario setup
    scenario = Scenario(
        classifier.configspace,
        n_trials=200,  # trials per SMAC run
        seed=seed,
        name=f"torch_distilbert_run_v2_{run_id}",
    )
    
    # Initial design with 5 configurations
    
    initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)
    
    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=classifier.train,
        initial_design=initial_design,
        overwrite=True,
    )
    incumbent = smac.optimize()
    
    # Validate default and incumbent configurations
    default_cost = smac.validate(classifier.configspace.get_default_configuration(), seed=seed)
    incumbent_cost = smac.validate(incumbent, seed=seed)
    logging.info(f"Default cost: {default_cost}, Incumbent cost: {incumbent_cost}")
    
    best_config = incumbent.get_dictionary()
    print("Best configuration:", best_config)
    
    # Save results  in a json file
    params_output_dir = "/scratch/hpc-prf-eaitcm/rishi_thesis/new_optimised_hyperparameters_bert_v2"
    os.makedirs(params_output_dir, exist_ok=True)
    file_path = f"{params_output_dir}/new_smac_params_run_v2_{run_id}.json"
    results = {
        "run_id": run_id,
        "seed": seed,
        "best_configuration": best_config,
        "default_cost": default_cost,
        "incumbent_cost": incumbent_cost,
    }
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {file_path}")
    
    end_time = time.time()
    logging.info(f"Total run time: {(end_time - start_time)/60:.2f} minutes")
    logging.info("SMAC optimization completed.")
