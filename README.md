

Detecting AI-Generated Text with Lightweight Explainable Models
🎓 Master's Thesis Repository

This repository contains the research, code, and documentation for a Master's thesis focusing on the detection of AI-generated text using lightweight, explainable models. The core objective is to develop and evaluate methodologies for distinguishing human-written content from that produced by artificial intelligence, with an emphasis on interpretability and efficiency.

📂 Repository Structure

thesis/:

src/: Contains all source code for the project, categorized by functionality.

src/BERT_model/: Python scripts for fine-tuning, optimizing, and evaluating BERT-based models.

bert_ablation.py: Script for performing ablation studies on optimized BERT configurations.

bert_optimisation.py: Implements hyperparameter optimization for BERT models using SMAC.

finetuning_bert.py: Script for fine-tuning DistilBERT on sequence classification tasks.

inference.py: Utility for running inference and evaluating fine-tuned models on test datasets.

multi_hyperparams_finetuning.py: Script to sequentially run and evaluate multiple hyperparameter configurations.

src/run_perplexity.ipynb: A Jupyter Notebook for calculating text perplexity using various pre-trained models.

src/run_perplexity.py: A Python script version of the perplexity calculation, suitable for command-line execution.

data/: (Conceptual; not explicitly in your repomix-output.xml but often present in such projects)

Raw and processed datasets used for training, validation, and testing the models. This folder is not included in the provided repomix-output.xml but would typically reside here.

results/: (Conceptual; not explicitly in your repomix-output.xml but often present in such projects)

Output artifacts from experiments, including evaluation metrics, plots, logs, and saved model checkpoints. This folder is not included in the provided repomix-output.xml but would typically reside here.

🚀 Getting Started

To replicate the experiments or utilize the code, follow the instructions below.

📝 Prerequisites

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment:

code
Bash
download
content_copy
expand_less

python -m venv venv
source venv/bin/activate # On Windows: `venv\Scripts\activate`
pip install -r requirements.txt # (Assuming you will create a requirements.txt)

Note: A requirements.txt file detailing all necessary Python packages is crucial for reproducibility. It should include torch, transformers, scikit-learn, pandas, numpy, tqdm, tabulate, smac, evaluate, and matplotlib.

🔍 1. Run a Perplexity Test on Your Dataset

The run_perplexity.py script (or its Jupyter Notebook counterpart run_perplexity.ipynb) allows you to calculate the perplexity of texts in your dataset using a pre-trained language model (e.g., GPT-2). This can help in understanding the "naturalness" or "predictability" of human-generated versus AI-generated text.

Your dataset should be in CSV or JSON format and contain a text column (e.g., tweet, tweets) and a label column (e.g., label, artificial) where 0 indicates original/human text and 1 indicates fake/AI-generated text.

Using the Python script:

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
python src/run_perplexity.py --input path/to/your_dataset.csv/json --limit 100 --model_id gpt2

--input: Path to your dataset file (.csv or .json).

--limit: (Optional) Maximum number of tweets per class to process.

--model_id: (Optional) Hugging Face model ID to use for perplexity calculation (default: gpt2).

Using the Jupyter Notebook:

Open src/run_perplexity.ipynb in a Jupyter environment (e.g., Google Colab, local Jupyter Lab) and follow the instructions within the notebook to configure your input_file and limit.

🧪 2. Fine-tuning BERT Models

The src/BERT_model/ directory contains scripts for fine-tuning DistilBERT for sequence classification.

Single Run Fine-tuning:

To fine-tune the model with a specific configuration (e.g., as part of a run array):

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
python src/BERT_model/finetuning_bert.py --run_index 0 --seed 42 --data_path /path/to/dataset.jsonl

--run_index: Index of the run (useful for SLURM job arrays).

--seed: Random seed for reproducibility.

--data_path: Path to the dataset file (JSON or JSON Lines).

Multi-Hyperparameter Fine-tuning:

To evaluate multiple predefined hyperparameter configurations sequentially:

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
python src/BERT_model/multi_hyperparams_finetuning.py --seed 42 --train_file /path/to/train_data.json --val_file /path/to/val_data.json

--seed: Base random seed (seeds for individual configurations will be offset from this).

--train_file: Path to the training dataset (JSON Lines).

--val_file: Path to the validation dataset (JSON Lines).

⚙️ 3. Hyperparameter Optimization with SMAC

The src/BERT_model/bert_optimisation.py script uses SMAC to find optimal hyperparameters for DistilBERT. This script is designed to be run in environments like SLURM job arrays to parallelize the optimization process.

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
# Example for a SLURM environment (assuming SLURM_ARRAY_TASK_ID is set)
# export SLURM_ARRAY_TASK_ID=0 # For a single local run test
python src/BERT_model/bert_optimisation.py

Ensure the dataset_path variable within bert_optimisation.py is updated to point to your training data.

The output directory for optimized hyperparameters (params_output_dir) should also be configured.

📊 4. Model Inference and Evaluation

After fine-tuning, you can evaluate your models on various test datasets using the src/BERT_model/inference.py script.

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
python src/BERT_model/inference.py --model_dir /path/to/your_saved_models --test_dir /path/to/your_test_datasets

--model_dir: Directory containing your fine-tuned transformer models (e.g., best_model_run_0).

--test_dir: Directory containing your test dataset JSON files.

🤝 Contribution

This repository serves as a record of the Master's thesis work. For academic inquiries or potential collaborations, please refer to the contact information in the thesis document.

📄 License

This project is licensed under the Apache License, Version 2.0 - see the LICENSE file for details.