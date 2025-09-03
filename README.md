

# 🧠 Detecting AI-Generated Text with Lightweight Explainable Models

🎓 *Master’s Thesis Repository*

This repository contains the research, code, and documentation for a Master's thesis focused on **detecting AI-generated text** using **lightweight, explainable models**.
The goal is to develop methodologies that are both **efficient** and **interpretable**, distinguishing **human-written** from **AI-generated** text.

---

## 📂 Repository Structure

```bash
.
├── thesis/                     # Thesis-related documents
├── src/                        # Source code
│   ├── BERT_model/             # Fine-tuning & optimization scripts
│   │   ├── bert_ablation.py
│   │   ├── bert_optimisation.py
│   │   ├── finetuning_bert.py
│   │   ├── inference.py
│   │   └── multi_hyperparams_finetuning.py
│   ├── run_perplexity.ipynb    # Perplexity analysis (Jupyter)
│   └── run_perplexity.py       # Perplexity analysis (CLI)
└── README.md
```

---

## 🚀 Getting Started

### 📝 Prerequisites

* **Python 3.8+**
* Recommended: Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> 📦 Required packages include: `torch`, `transformers`, `scikit-learn`, `pandas`, `numpy`, `tqdm`, `tabulate`, `smac`, `evaluate`, `matplotlib`.

---

## 🔍 1. Run Perplexity Test

Calculate perplexity of texts (human vs. AI-generated) using pre-trained models (default: `gpt2`).

```bash
python src/run_perplexity.py \
  --input path/to/dataset.csv \
  --limit 100 \
  --model_id gpt2
```

* `--input` : Path to dataset (`.csv` or `.json`)
* `--limit` : (Optional) Max samples per class
* `--model_id` : Hugging Face model (default: `gpt2`)

📓 Or use Jupyter: open `src/run_perplexity.ipynb` in Colab or Jupyter Lab.

---

## 🧪 2. Fine-tune BERT Models

### Single-run Fine-tuning

```bash
python src/BERT_model/finetuning_bert.py \
  --run_index 0 \
  --seed 42 \
  --data_path /path/to/dataset.jsonl
```

### Multi-hyperparameter Fine-tuning

```bash
python src/BERT_model/multi_hyperparams_finetuning.py \
  --seed 42 \
  --train_file /path/to/train.json \
  --val_file /path/to/val.json
```

---

## ⚙️ 3. Hyperparameter Optimization (SMAC)

```bash
python src/BERT_model/bert_optimisation.py
```

> 💡 Configure `dataset_path` and `params_output_dir` inside the script before running.

---

## 📊 4. Model Inference & Evaluation

```bash
python src/BERT_model/inference.py \
  --model_dir /path/to/saved_models \
  --test_dir /path/to/test_data
```

---

## 🤝 Contribution

This repository documents the **Master's thesis work**.
For academic inquiries or collaborations, please refer to the author.

---

## 📄 License

Licensed under the **Apache License, Version 2.0**.
See the [LICENSE](LICENSE) file for details.
