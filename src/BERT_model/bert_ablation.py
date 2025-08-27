"""This ablation script calls the train function from the optimisatio script. This script implements greedy ablation approach"""

import json
import sys
import os
import logging
from BERT_optimisation.torch_optimise_bert import BertClassifier
import torch
import matplotlib.pyplot as plt
import numpy as np

# Setup logging to print messages to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class DictConfig:
    """
    Simple wrapper so we can pass a plain dict as if it were a SMAC Configuration.
    Provides get_dictionary() for compatibility with BertClassifier.train.
    """
    def __init__(self, cfg: dict):
        self._cfg = cfg
    def get_dictionary(self):
        return self._cfg


def train_cv(config: dict, seeds=[60,61,62]):
    """
    Run cross-validation by training the model with multiple seeds and averaging the validation loss.
    Wraps the plain dict into DictConfig so .train() sees .get_dictionary().
    """
    logger.info("Starting train_cv with config: %s, seeds: %s", config, seeds)
    classifier = BertClassifier()
    wrapped_config = DictConfig(config)

    losses = []
    for seed in seeds:
        logger.debug("Training with seed: %d", seed)
        torch.manual_seed(seed)
        loss = classifier.train(wrapped_config, seed=seed)
        logger.debug("Validation loss for seed %d: %.4f", seed, loss)
        losses.append(loss)

    avg_loss = sum(losses) / len(losses)
    logger.info("Average validation loss across %d seeds: %.4f", len(seeds), avg_loss)
    return avg_loss


def run_ablation(source, target, train_fn, output_dir="ablation_results"):
    """
    Perform ablation study by flipping hyperparameters from source (optimized) to target (default) configuration.
    Saves results to JSON files, generates plots, and reports percentage change in validation loss.
    """
    logger.info("Starting ablation study with output directory: %s", output_dir)
    logger.debug("Source config: %s", source)
    logger.debug("Target config: %s", target)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Created output directory: %s", output_dir)

    path = []
    history = []
    flippable = {}
    parameters = list(set(source.keys()) & set(target.keys()))
    logger.debug("Parameters considered: %s", parameters)

    # Identify hyperparameters that differ between source and target
    for param in parameters:
        if source[param] != target[param]:
            flippable[param] = (source[param], target[param])
    logger.info("Flippable hyperparameters: %s", flippable)

    # Evaluate source (optimized) configuration
    current_config = source.copy()
    logger.info("Evaluating source (optimized) configuration: %s", current_config)
    source_loss = train_fn(current_config)
    path.append(("source (optimized)", source_loss))
    history.append({
        "config": current_config.copy(),
        "validation_loss": source_loss
    })
    logger.info("Source validation loss: %.4f", source_loss)

    # Ablation loop
    while flippable:
        performances = {}
        for param, (_, new_value) in flippable.items():
            config = current_config.copy()
            config[param] = new_value
            logger.debug("Evaluating config with %s flipped to %s", param, new_value)
            val_loss = train_fn(config)
            performances[param] = val_loss
            history.append({"config": config.copy(), "validation_loss": val_loss})
            # Save history to JSON
            with open(f"{output_dir}/ablation_{param}.json", "w") as fh:
                json.dump(history, fh, indent=4)

        # Select best flip
        best_param = min(performances, key=performances.get)
        best_loss = performances[best_param]
        path.append((f"{best_param} (to default)", best_loss))
        logger.info("Selected %s with validation loss: %.4f", best_param, best_loss)
        # Update current config
        current_config[best_param] = flippable[best_param][1]
        del flippable[best_param]

    # Save final results
    with open(f"{output_dir}/ablation_path.json", "w") as fh:
        json.dump({"path": path, "history": history}, fh, indent=4)
    logger.info("Saved ablation path to %s/ablation_path.json", output_dir)

    # Plot ablation path
    logger.info("Generating ablation path plot")
    plt.figure(figsize=(8, 6))
    steps = [step[0] for step in path]
    losses = [step[1] for step in path]
    plt.plot(steps, losses, marker='o', linestyle='-')
    plt.xlabel('Ablation Step (Flipping to Default)')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs. Ablation Path')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_path_plot.png")
    plt.close()
    logger.info("Saved ablation path plot to %s/ablation_path_plot.png", output_dir)

    # Impact calculations
    source_loss = path[0][1]
    impact_results = []
    for record in history:
        cfg = record['config']
        for param in target:
            if cfg.get(param) == target[param] and all(cfg.get(p) == source[p] for p in source if p != param):
                absolute_change = record['validation_loss'] - source_loss
                percent_change = (absolute_change / source_loss * 100) if source_loss else 0
                impact_results.append({
                    'hyperparameter': param,
                    'absolute_change': absolute_change,
                    'percentage_change': percent_change
                })
                break

    with open(f"{output_dir}/ablation_impact.json", "w") as fh:
        json.dump(impact_results, fh, indent=4)
    logger.info("Saved impact results to %s/ablation_impact.json", output_dir)

    # Plot impacts
    names = [r['hyperparameter'] for r in impact_results]
    impacts = [r['absolute_change'] for r in impact_results]
    percents = [r['percentage_change'] for r in impact_results]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(names, impacts)
    plt.xlabel('Hyperparameter')
    plt.ylabel('Change in Validation Loss')
    plt.title('Impact of Reverting Hyperparameters')
    for bar, pct in zip(bars, percents):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{pct:.2f}%", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_impact_plot.png")
    plt.close()
    logger.info("Saved impact plot to %s/ablation_impact_plot.png", output_dir)

    logger.info("Ablation study completed")
    return path, history


if __name__ == "__main__":
    logger.info("Script Started ")
    
    
    # target config
    default_config = {
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "batch_size": 32,
        "optimizer": "adamw"
    }
    
    # Source config (baseline)
    optimized_config = {
        "learning_rate": 3.49886998e-05,
        "weight_decay": 0.1678043365548,
        "batch_size": 37,
        "optimizer": "adamw"
    }
    run_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    output_dir = f"<your/output/directory/path>_run_{run_id}"
    path, history = run_ablation(optimized_config, default_config, train_cv, output_dir)
