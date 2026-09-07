# Clickbait Detection Using XLM-RoBERTa and BiLSTM

A Streamlit-based clickbait detection application that compares a traditional BiLSTM model with an attention-enhanced hybrid model using XLM-RoBERTa.

## Overview

The system detects whether a news headline is clickbait or non-clickbait using two models:

- **Baseline BiLSTM** — Uses a trainable 300-dimensional embedding layer and BiLSTM.
- **Proposed Model** — Uses frozen XLM-RoBERTa contextual embeddings, BiLSTM, and residual self-attention.

The application displays the prediction and confidence score of both models for comparison.

## Features

| Feature             | Description                                      |
| ------------------- | ------------------------------------------------ |
| Headline Detection  | Classifies a provided news headline              |
| Two Models          | Compares the baseline and proposed architectures |
| Confidence Score    | Displays prediction confidence                   |
| Model Comparison    | Shows both model predictions side by side        |
| XLM-RoBERTa         | Provides multilingual contextual representations |
| Streamlit Interface | Simple web-based interface                       |

## Detection Pipeline

```text
News Headline
      │
      ▼
Text Preprocessing
      │
      ├───────────────┐
      ▼               ▼
 Baseline          Proposed
  BiLSTM          XLM-RoBERTa
      │               │
      │             BiLSTM
      │               │
      │        Residual Self-Attention
      │               │
      └───────┬───────┘
              ▼
       Classification
              │
              ▼
   Clickbait / Non-clickbait
```

## Dataset

The models were trained using the CLICK-ID dataset.

| Property        | Details               |
| --------------- | --------------------- |
| Dataset         | CLICK-ID              |
| Total Headlines | 15,000                |
| Clickbait       | 6,290                 |
| Non-clickbait   | 8,710                 |
| Language        | Indonesian            |
| Task            | Binary Classification |

Dataset link: https://data.mendeley.com/datasets/k42j7x2kpn/1

## Tech Stack

- Python 3.11
- Streamlit
- TensorFlow / Keras
- PyTorch
- Transformers
- XLM-RoBERTa
- NumPy
- Pandas
- Plotly
- Git / GitHub

## Project Structure

```text
clickbait-detection-app/
├── app.py
├── requirements.txt
├── .gitignore
│
├── deployment_baseline/
│   ├── baseline_bilstm.keras
│   ├── baseline_tokenizer.pkl
│   └── baseline_config.pkl
│
└── deployment_proposed/
    ├── xlmr_bilstm_attention.keras
    └── xlmr_config.txt
```

## Setup

### Prerequisites

- Python 3.11
- Git

### Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd clickbait-detection-app
```

Create and activate a virtual environment:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate   # On Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

The application will open in your browser through the local Streamlit server.

## Model Performance

Current experimental results:

| Model                      | Accuracy | Precision | Recall | F1-Score |
| -------------------------- | -------- | --------- | ------ | -------- |
| Baseline BiLSTM            | 0.7461   | 0.7030    | 0.6830 | 0.6929   |
| BiLSTM + XLM-R             | 0.7651   | 0.7183    | 0.7235 | 0.7209   |
| BiLSTM + XLM-R + Attention | 0.7855   | 0.7705    | 0.6956 | 0.7311   |

## Authors

- Christian James Cahilig
- [Member Name]
- [Member Name]
- [Member Name]

University of Mindanao
Bachelor of Science in Computer Science
Thesis Project

## License

This project is developed for academic and educational purposes.
