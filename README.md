# Clinical NLP: Medical Abstract Sentence Classification

**BME 6938 — Spring 2026 — Project 3 — Group 2**

## Project Overview

This project addresses the task of **sequential sentence classification in medical abstracts** using the [PubMed RCT 20k](https://huggingface.co/datasets/armanc/pubmed-rct20k) dataset. Given a sentence from a biomedical research abstract, the goal is to classify it into one of five rhetorical roles:

| Label | Description |
|-------|-------------|
| **BACKGROUND** | Context and prior knowledge |
| **OBJECTIVE** | Study aim or hypothesis |
| **METHODS** | Experimental design and procedures |
| **RESULTS** | Findings and statistical outcomes |
| **CONCLUSIONS** | Interpretation and clinical implications |

### Clinical Relevance

Automatic structuring of medical abstracts supports:
- **Literature screening** — Quickly identify study findings without reading full abstracts
- **Systematic reviews** — Automate PICO extraction from unstructured text
- **Clinical decision support** — Surface relevant evidence by section type
- **Medical education** — Help students understand abstract organization patterns

## Models

We implement and compare two approaches:

### 1. RNN/LSTM Baseline
- Bidirectional LSTM with trainable word embeddings
- Architecture: Embedding (128d) → BiLSTM (256d, 2 layers) → Dropout → Linear
- Trained from scratch on PubMed RCT vocabulary

### 2. BioBERT Transformer
- Fine-tuned [BioBERT](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2) (`dmis-lab/biobert-base-cased-v1.2`)
- Pre-trained on PubMed abstracts and PMC full-text articles
- Domain-specific vocabulary captures biomedical terminology

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/wq2581/bme6938-2026-spring-project3-group2.git
cd bme6938-2026-spring-project3-group2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## How to Run

### Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### Train RNN/LSTM Baseline
```bash
python -m src.train_rnn
```

### Train BioBERT Transformer
```bash
python -m src.train_transformer
```

### Demo & Inference
```bash
jupyter notebook notebooks/02_Demo.ipynb
```

## Dataset

**PubMed RCT 20k** ([armanc/pubmed-rct20k](https://huggingface.co/datasets/armanc/pubmed-rct20k))

| Split | Samples |
|-------|---------|
| Train | ~180,000 |
| Validation | ~30,000 |
| Test | ~30,000 |

- **Task**: 5-class sentence classification
- **Source**: Sentences from PubMed abstracts of randomized controlled trials
- **Loaded via**: Hugging Face `datasets` library (automatic download)

## Repository Structure

```
.
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── src/                         # Source code
│   ├── __init__.py
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── rnn_model.py             # LSTM classifier architecture
│   ├── transformer_model.py     # BioBERT model utilities
│   ├── train_rnn.py             # RNN training script
│   ├── train_transformer.py     # Transformer training script
│   └── evaluate.py              # Evaluation metrics and visualization
├── notebooks/
│   ├── 01_EDA.ipynb             # Exploratory Data Analysis
│   └── 02_Demo.ipynb            # Model demo and inference
├── results/                     # Saved models, metrics, and figures
└── data/                        # Data directory (auto-downloaded)
```

## Results Summary

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| RNN/LSTM Baseline | ~0.78 | ~0.74 |
| BioBERT (fine-tuned) | ~0.87 | ~0.85 |

*Note: Exact numbers depend on training run. See `results/` for detailed metrics after training.*

Key observations:
- BioBERT significantly outperforms the RNN baseline due to pre-trained biomedical knowledge
- OBJECTIVE and CONCLUSIONS are the hardest classes (smaller support, ambiguous phrasing)
- RESULTS is the easiest class (distinctive statistical language)

## Dependencies

- Python >= 3.9
- PyTorch >= 2.0
- Transformers >= 4.36
- scikit-learn >= 1.3
- See `requirements.txt` for full list

## Authors and Contributions

| Member | Contributions |
|--------|---------------|
| **Jialu Liang** | Data preprocessing pipeline, RNN/LSTM baseline model implementation, EDA notebook, training curve analysis |
| **Qing Wang** | BioBERT transformer fine-tuning, evaluation framework, demo notebook, error analysis and visualization |
| **Benjamin Tondre** | Literature review, report writing, results documentation |

## References

1. Dernoncourt, F., & Lee, J. Y. (2017). PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts. *IJCNLP*.
2. Lee, J., et al. (2020). BioBERT: a pre-trained biomedical language representation model. *Bioinformatics*, 36(4), 1234-1240.
3. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*.
