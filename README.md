# Detecting AI-Generated Text with Lightweight Explainable Models

## 🎓 Master's Thesis Repository
This repository contains the research, code, and documentation for my master's thesis submitted at [University Name].

## 📂 Structure

- `thesis/` folder contains the final compiled version of my master's thesis:
    - `thesis.pdf`: Full thesis document
    - `presentation.pdf`: Slides used during thesis defense

- `code/`: Source code for model training and experiments
- `data/`: Raw and processed datasets
- `results/`: Plots, logs, and evaluation results

## 🚀 Getting Started

### 🔍 1. Run a Perplexity Test on Your Dataset

Use the provided script to test the perplexity of your dataset using a pretrained model. The dataset should be labelled and should be wither in csv or json format:

```bash
python code/run_perplexity.py --input path/to/your_dataset.csv/json


