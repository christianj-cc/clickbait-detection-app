import streamlit as st
import numpy as np
import pickle
import os
import re
import time
import textwrap

import tensorflow as tf
from transformers import AutoTokenizer, AutoModel
import torch


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Thesis Clickbait Detection",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# MARKDOWN HELPER
# =============================================================================
# IMPORTANT: CommonMark ends a raw <div>/<span> HTML block at the first
# blank line and starts re-parsing whatever follows as plain Markdown
# (which is why nested tags were showing up as literal text/code before).
# This helper joins every line with a single "\n" and never inserts an
# empty line, so the whole block stays one continuous raw-HTML block.
# <style>, <pre>, and <script> are exempt from this rule (browsers treat
# them as raw-text elements), so CSS blocks and the <pre> diagrams below
# are free to contain blank lines / whitespace-sensitive formatting.

def md(*lines):
    st.markdown("\n".join(str(l) for l in lines if l != ""), unsafe_allow_html=True)


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary: #0A0E1A;
    --bg-secondary: #0F1629;
    --surface: #131D35;
    --surface-raised: #1A2542;

    --border: rgba(56, 189, 248, 0.12);
    --border-bright: rgba(56, 189, 248, 0.35);

    --text-primary: #E2E8F0;
    --text-secondary: #94A3B8;
    --text-muted: #475569;

    --accent: #38BDF8;
    --accent-dark: #0EA5E9;
    --accent-dim: rgba(56, 189, 248, 0.10);

    --success: #34D399;
    --success-dim: rgba(52, 211, 153, 0.08);

    --error: #F87171;
    --error-dim: rgba(248, 113, 113, 0.08);

    --warning: #FBBF24;
    --warning-dim: rgba(251, 191, 36, 0.08);
}

html, body, .stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

.block-container {
    padding-top: 0.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] > div {
    padding: 1.5rem 1.25rem;
}

.sidebar-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}

.logo-icon {
    width: 34px;
    height: 34px;
    flex-shrink: 0;
    background: linear-gradient(135deg, var(--accent-dark), var(--accent));
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Space Mono', monospace;
    font-size: 15px;
    font-weight: 700;
    color: var(--bg-primary);
    box-shadow: 0 0 16px rgba(56,189,248,0.25);
}

.brand-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    line-height: 1.4;
}

.brand-name span {
    color: var(--accent);
    display: block;
    font-size: 0.8rem;
}

.sidebar-section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

.status-pill {
    display: flex;
    align-items: center;
    gap: 9px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.85rem;
    margin-bottom: 0.75rem;
}

.status-dot {
    width: 7px;
    height: 7px;
    background: var(--success);
    border-radius: 50%;
    box-shadow: 0 0 7px var(--success);
}

.status-text {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-secondary);
}

/* ---------------------------------------------------------------------- */
/* COLLAPSIBLE MODEL / BENCHMARK DROPDOWNS                                */
/* ---------------------------------------------------------------------- */

details.model-detail {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--border-bright);
    border-radius: 8px;
    margin-bottom: 0.65rem;
    overflow: hidden;
}

details.model-detail summary {
    cursor: pointer;
    list-style: none;
    padding: 0.65rem 0.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-secondary);
}

details.model-detail summary::-webkit-details-marker {
    display: none;
}

details.model-detail summary::after {
    content: '▸';
    font-size: 0.7rem;
    transition: transform 0.15s ease;
    color: var(--text-muted);
}

details.model-detail[open] summary::after {
    transform: rotate(90deg);
}

details.model-detail.baseline {
    border-left-color: var(--text-muted);
}

details.model-detail.baseline summary {
    color: var(--text-secondary);
}

details.model-detail.proposed {
    border-left-color: var(--accent);
    box-shadow: 0 0 14px rgba(56,189,248,0.08);
}

details.model-detail.proposed summary {
    color: var(--accent);
}

details.model-detail .detail-body {
    padding: 0 0.8rem 0.85rem 0.8rem;
    border-top: 1px solid var(--border);
    padding-top: 0.7rem;
}

.arch-terminal {
    background: #05070D;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.1rem 0.75rem;
    margin-top: 0.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    line-height: 1.75;
    color: #E5E7EB;
    text-align: center;
}

.arch-terminal .arrow {
    color: var(--text-muted);
}

.arch-terminal .final-line {
    color: var(--accent);
    font-weight: 700;
}

.spec-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 0.5rem;
}

.spec-chip {
    background: var(--accent-dim);
    border: 1px solid var(--border-bright);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 0.5rem;
    padding: 0.2rem 0.5rem;
    border-radius: 99px;
    letter-spacing: 0.02em;
    white-space: nowrap;
}

.bench-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Space Mono', monospace;
    font-size: 0.55rem;
    margin-top: 0.5rem;
}

.bench-table th, .bench-table td {
    padding: 0.4rem 0.3rem;
    text-align: center;
    border-bottom: 1px solid var(--border);
    color: var(--text-secondary);
}

.bench-table th {
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.46rem;
}

.bench-table tr:last-child td {
    color: var(--accent);
    font-weight: 700;
}

/* ---------------------------------------------------------------------- */
/* HEADER                                                                  */
/* ---------------------------------------------------------------------- */

.main-header {
    text-align: center;
    padding: 2.75rem 2rem 1.5rem;
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 400px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
}

.header-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.header-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
    line-height: 1.15;
    letter-spacing: -0.02em;
}

.header-title span {
    color: var(--accent);
}

.header-sub {
    font-size: 0.95rem;
    color: var(--text-secondary);
    margin-top: 0.75rem;
    font-weight: 300;
}

/* ---------------------------------------------------------------------- */
/* STEPS                                                                   */
/* ---------------------------------------------------------------------- */

.step-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin: 1.5rem auto 2rem auto;
    max-width: 520px;
}

.step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 5px;
    flex: 1;
}

.step-dot {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    border: 1.5px solid var(--border-bright);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-muted);
    background: var(--surface);
}

.step-dot.active {
    border-color: var(--accent);
    color: var(--accent);
    box-shadow: 0 0 10px rgba(56,189,248,0.3);
}

.step-dot.done {
    border-color: var(--success);
    color: var(--success);
    background: rgba(52,211,153,0.08);
    box-shadow: 0 0 8px rgba(52,211,153,0.2);
}

.step-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.52rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    text-align: center;
}

.step-label.active {
    color: var(--accent);
}

.step-label.done {
    color: var(--success);
}

.step-connector {
    height: 1px;
    flex: 1;
    max-width: 60px;
    background: var(--border);
    margin-bottom: 18px;
}

.step-connector.done {
    background: var(--success);
    opacity: 0.5;
}

/* ---------------------------------------------------------------------- */
/* INPUT                                                                   */
/* ---------------------------------------------------------------------- */

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
}

.headline-panel {
    background: var(--surface);
    border: 1px solid var(--border-bright);
    border-radius: 12px;
    padding: 1.25rem;
    transition: 0.2s ease;
}

.headline-panel:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 20px rgba(56,189,248,0.10);
}

textarea {
    color: var(--text-primary) !important;
}

[data-testid="stTextArea"] textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    color: var(--text-primary) !important;
    resize: vertical !important;
}

[data-testid="stTextArea"] textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ---------------------------------------------------------------------- */
/* BUTTON                                                                  */
/* ---------------------------------------------------------------------- */

.stButton > button {
    background: linear-gradient(135deg, var(--accent-dark), var(--accent)) !important;
    color: var(--bg-primary) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.8rem 1.5rem !important;
    box-shadow: 0 4px 20px rgba(56,189,248,0.3) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(56,189,248,0.45) !important;
}

/* ---------------------------------------------------------------------- */
/* RESULT CARDS                                                           */
/* ---------------------------------------------------------------------- */

.results-heading {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    text-align: center;
    margin-bottom: 1.25rem;
}

.model-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    min-height: 315px;
}

.model-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
}

.model-card.baseline {
    border-color: rgba(148,163,184,0.18);
}

.model-card.proposed {
    border-color: rgba(56,189,248,0.28);
    box-shadow: 0 0 25px rgba(56,189,248,0.06);
}

.model-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: var(--text-muted);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.model-title {
    font-size: 1.05rem;
    font-weight: 500;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.model-title span {
    color: var(--accent);
}

.verdict {
    font-family: 'Space Mono', monospace;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-top: 1.25rem;
    margin-bottom: 0.35rem;
}

.verdict-clickbait {
    color: var(--error);
}

.verdict-normal {
    color: var(--success);
}

.verdict-sub {
    color: var(--text-secondary);
    font-size: 0.82rem;
    font-weight: 300;
}

/* ---------------------------------------------------------------------- */
/* CONFIDENCE                                                              */
/* ---------------------------------------------------------------------- */

.confidence-panel {
    margin-top: 1.5rem;
}

.conf-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 0.7rem;
}

.conf-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
}

.conf-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
}

.conf-track {
    position: relative;
    height: 8px;
    border-radius: 99px;
    overflow: hidden;
    background: rgba(71,85,105,0.15);
}

.conf-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}

.conf-scale {
    display: flex;
    justify-content: space-between;
    margin-top: 5px;
    font-family: 'Space Mono', monospace;
    font-size: 0.48rem;
    color: var(--text-muted);
}

/* ---------------------------------------------------------------------- */
/* INFERENCE TIME STAT ROW                                                 */
/* ---------------------------------------------------------------------- */

.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-top: 0.9rem;
    padding-top: 0.7rem;
    border-top: 1px dashed var(--border);
}

.stat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.55rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-secondary);
}

/* ---------------------------------------------------------------------- */
/* AGREEMENT PANEL                                                         */
/* ---------------------------------------------------------------------- */

.comparison-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-top: 1.25rem;
    text-align: center;
}

.comparison-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    color: var(--text-muted);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.comparison-value {
    font-size: 0.9rem;
    color: var(--text-secondary);
}

/* ---------------------------------------------------------------------- */
/* HEADLINE DISPLAY                                                        */
/* ---------------------------------------------------------------------- */

.analyzed-headline {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
}

.analyzed-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}

.analyzed-text {
    font-size: 0.9rem;
    line-height: 1.5;
    color: var(--text-primary);
}

/* ---------------------------------------------------------------------- */
/* FOOTER                                                                  */
/* ---------------------------------------------------------------------- */

.footer-text {
    text-align: center;
    color: var(--text-muted);
    padding: 2.5rem 0 1rem;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.08em;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}

hr {
    border-color: var(--border) !important;
}

</style>
""", unsafe_allow_html=True)


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.join(BASE_DIR, "deployment_baseline")
PROPOSED_DIR = os.path.join(BASE_DIR, "deployment_proposed")

BASELINE_MODEL_PATH = os.path.join(BASELINE_DIR, "baseline_bilstm.keras")
BASELINE_TOKENIZER_PATH = os.path.join(BASELINE_DIR, "baseline_tokenizer.pkl")
BASELINE_CONFIG_PATH = os.path.join(BASELINE_DIR, "baseline_config.pkl")

PROPOSED_MODEL_PATH = os.path.join(PROPOSED_DIR, "xlmr_bilstm_attention.keras")
PROPOSED_CONFIG_PATH = os.path.join(PROPOSED_DIR, "xlmr_config.txt")


# =============================================================================
# ARCHITECTURE DIAGRAMS (rendered inside <pre>, so whitespace is preserved)
# =============================================================================

def build_arch_html(lines):
    """
    Renders an architecture diagram as individually centered lines
    joined with <br>, instead of relying on <pre>/white-space:pre —
    Streamlit's markdown pipeline doesn't reliably preserve raw
    multi-line whitespace inside nested custom HTML, which is what
    was collapsing the diagrams into wrapped paragraphs.
    """
    html_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped in ("CLICKBAIT", "NON-CLICKBAIT", "CLICKBAIT / NON-CLICKBAIT"):
            html_lines.append(f'<span class="final-line">{stripped}</span>')
        elif stripped in ("│", "▼", "┌────┴────┐", "└────┬────┘") or set(stripped) <= set("│▼┌┴┐└┬┘─ "):
            html_lines.append(f'<span class="arrow">{stripped}</span>' if stripped else "")
        else:
            html_lines.append(stripped)
    return "<br>".join(html_lines)


BASELINE_ARCH_HTML = build_arch_html([
    "INPUT HEADLINE",
    "│",
    "▼",
    "Tokenization",
    "│",
    "▼",
    "300D Trainable",
    "Embedding Layer",
    "│",
    "▼",
    "Bidirectional LSTM",
    "64 Units",
    "│",
    "▼",
    "Dropout",
    "0.20",
    "│",
    "▼",
    "Batch Normalization",
    "│",
    "▼",
    "Global Max Pooling 1D",
    "│",
    "▼",
    "Dense Layer",
    "Sigmoid (1 unit)",
    "│",
    "▼",
    "CLICKBAIT / NON-CLICKBAIT",
])

PROPOSED_ARCH_HTML = build_arch_html([
    "INPUT HEADLINE",
    "│",
    "▼",
    "XLM-RoBERTa Tokenizer",
    "│",
    "▼",
    "XLM-RoBERTa Base",
    "Frozen Encoder",
    "│",
    "▼",
    "768D Token Embeddings",
    "│",
    "▼",
    "Bidirectional LSTM",
    "64 Units",
    "│",
    "┌────┴────┐",
    "│",
    "▼",
    "Self-Attention",
    "│",
    "▼",
    "Attention Output",
    "│",
    "└────┬────┘",
    "▼",
    "Residual Addition",
    "BiLSTM + Attention Output",
    "│",
    "▼",
    "Layer Normalization",
    "│",
    "▼",
    "Dropout",
    "0.20",
    "│",
    "▼",
    "Batch Normalization",
    "│",
    "▼",
    "Global Max Pooling 1D",
    "│",
    "▼",
    "Dense Layer",
    "Sigmoid (1 unit)",
    "│",
    "▼",
    "CLICKBAIT / NON-CLICKBAIT",
])


# =============================================================================
# BENCHMARK RESULTS
# =============================================================================

BENCHMARK_ROWS = [
    ("Baseline BiLSTM", "0.7461", "0.7030", "0.6830", "0.6929"),
    ("BiLSTM + XLM-R", "0.7651", "0.7183", "0.7235", "0.7209"),
    ("BiLSTM + XLM-R + Attn", "0.7855", "0.7705", "0.6956", "0.7311"),
]


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_text(text):
    """
    Same preprocessing used by the baseline training pipeline.
    """
    text = str(text).lower()
    text = text.replace("!", " tanda seru ")
    text = text.replace("?", " tanda tanya ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =============================================================================
# LOAD BASELINE MODEL
# =============================================================================

@st.cache_resource
def load_baseline():
    model = tf.keras.models.load_model(BASELINE_MODEL_PATH)

    with open(BASELINE_TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(BASELINE_CONFIG_PATH, "rb") as f:
        config = pickle.load(f)

    return model, tokenizer, config


# =============================================================================
# LOAD XLM-R
# =============================================================================

@st.cache_resource
def load_xlmr():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    encoder = AutoModel.from_pretrained("xlm-roberta-base")
    encoder.eval()

    for param in encoder.parameters():
        param.requires_grad = False

    return tokenizer, encoder


# =============================================================================
# LOAD PROPOSED MODEL
# =============================================================================

@st.cache_resource
def load_proposed():
    model = tf.keras.models.load_model(PROPOSED_MODEL_PATH)
    config = {}

    if os.path.exists(PROPOSED_CONFIG_PATH):
        with open(PROPOSED_CONFIG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    config[key.strip()] = value.strip()

    return model, config


# =============================================================================
# XLM-R FEATURE EXTRACTION
# =============================================================================

def extract_xlmr_embedding(text, tokenizer, encoder, max_length=128):
    encoded = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = encoder(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"]
        )

    embedding = outputs.last_hidden_state.cpu().numpy()
    return embedding.astype(np.float32)


# =============================================================================
# BASELINE PREDICTION
# =============================================================================

def predict_baseline(text, model, tokenizer, config):
    processed = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed])

    max_length = int(
        config.get("max_length", max(len(x) for x in sequence) if sequence else 1)
    )

    padded = tf.keras.utils.pad_sequences(
        sequence,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    probability = float(model.predict(padded, verbose=0)[0][0])
    prediction = int(probability >= 0.5)

    return prediction, probability


# =============================================================================
# PROPOSED MODEL PREDICTION
# =============================================================================

def predict_proposed(text, model, tokenizer, encoder, config):
    processed = preprocess_text(text)
    max_length = int(config.get("MAX_LENGTH", 128))

    embedding = extract_xlmr_embedding(processed, tokenizer, encoder, max_length)

    probability = float(model.predict(embedding, verbose=0)[0][0])
    prediction = int(probability >= 0.5)

    return prediction, probability


# =============================================================================
# STEP RENDERER
# =============================================================================

def render_steps(active_step=1):
    labels = ["Headline", "Analysis", "Results"]
    parts = ['<div class="step-row">']

    for i, label in enumerate(labels):
        num = i + 1

        if num < active_step:
            dot_class, label_class, inner = "done", "done", "✓"
        elif num == active_step:
            dot_class, label_class, inner = "active", "active", str(num)
        else:
            dot_class, label_class, inner = "", "", str(num)

        parts.append(
            f'<div class="step-item"><div class="step-dot {dot_class}">{inner}</div>'
            f'<div class="step-label {label_class}">{label}</div></div>'
        )

        if i < len(labels) - 1:
            connector_class = "done" if num < active_step else ""
            parts.append(f'<div class="step-connector {connector_class}"></div>')

    parts.append('</div>')
    md(*parts)


# =============================================================================
# RESULT CARD
# =============================================================================

def render_model_card(model_type, title, prediction, probability, inference_time, spec_chips):
    if prediction == 1:
        verdict = "CLICKBAIT"
        verdict_class = "verdict-clickbait"
        subtitle = "The headline is classified as clickbait."
        bar_color = "#F87171"
        confidence = probability
    else:
        verdict = "NON-CLICKBAIT"
        verdict_class = "verdict-normal"
        subtitle = "The headline is classified as non-clickbait."
        bar_color = "#34D399"
        confidence = 1.0 - probability

    confidence_pct = confidence * 100
    card_class = "proposed" if model_type == "proposed" else "baseline"
    chips_html = "".join(f'<span class="spec-chip">{chip}</span>' for chip in spec_chips)

    md(
        f'<div class="model-card {card_class}">',
        f'<div class="model-name">{model_type}</div>',
        f'<div class="model-title">{title}</div>',
        f'<div class="spec-chip-row">{chips_html}</div>',
        f'<div class="verdict {verdict_class}">{verdict}</div>',
        f'<div class="verdict-sub">{subtitle}</div>',
        '<div class="confidence-panel">',
        '<div class="conf-header">',
        '<span class="conf-title">Confidence</span>',
        f'<span class="conf-value" style="color:{bar_color};">{confidence_pct:.1f}%</span>',
        '</div>',
        '<div class="conf-track">',
        f'<div class="conf-fill" style="width:{confidence_pct}%;background:linear-gradient(90deg,{bar_color}88,{bar_color});box-shadow:0 0 12px {bar_color}55;"></div>',
        '</div>',
        '<div class="conf-scale"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>',
        '</div>',
        '<div class="stat-row">',
        '<span class="stat-label">Inference Time</span>',
        f'<span class="stat-value">{inference_time * 1000:.1f} ms</span>',
        '</div>',
        '</div>'
    )


# =============================================================================
# HEADER
# =============================================================================

md(
    '<div class="main-header">',
    '<div class="header-eyebrow">Natural Language Classification</div>',
    '<h1 class="header-title">Headline Clickbait <span>Detection</span></h1>',
    '<p class="header-sub">Compare a baseline BiLSTM with an XLM-RoBERTa enhanced attention model for news headline classification</p>',
    '</div>'
)


# =============================================================================
# LOAD MODELS
# =============================================================================

try:
    with st.spinner("Initializing detection models..."):
        baseline_model, baseline_tokenizer, baseline_config = load_baseline()
        proposed_model, proposed_config = load_proposed()
        xlmr_tokenizer, xlmr_encoder = load_xlmr()

    models_loaded = True

except Exception as e:
    models_loaded = False
    st.error(f"Unable to initialize the models: {e}")


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    md(
        '<div class="sidebar-header">',
        '<div class="logo-icon">◈</div>',
        '<div class="brand-name">HEADLINE CLICKBAIT<span>DETECTION</span></div>',
        '</div>'
    )

    md('<div class="sidebar-section-label">Detection System</div>')

    if models_loaded:
        md(
            '<div class="status-pill">',
            '<div class="status-dot"></div>',
            '<div class="status-text">SYSTEM READY</div>',
            '</div>'
        )
    else:
        md(
            '<div class="status-pill">',
            '<div class="status-dot" style="background:#F87171;box-shadow:0 0 7px #F87171;"></div>',
            '<div class="status-text">MODEL ERROR</div>',
            '</div>'
        )

    md('<div class="sidebar-section-label" style="margin-top:1.25rem;">Model Architectures</div>')

    md(
        '<details class="model-detail baseline">',
        '<summary>Baseline · BiLSTM</summary>',
        '<div class="detail-body">',
        '<div class="spec-chip-row">',
        '<span class="spec-chip">300D Trainable Embedding</span>',
        '<span class="spec-chip">BiLSTM · 64 units</span>',
        '<span class="spec-chip">No Attention</span>',
        '</div>',
        f'<div class="arch-terminal">{BASELINE_ARCH_HTML}</div>',
        '</div>',
        '</details>'
    )

    md(
        '<details class="model-detail proposed">',
        '<summary>Proposed · XLM-R + BiLSTM + Attn</summary>',
        '<div class="detail-body">',
        '<div class="spec-chip-row">',
        '<span class="spec-chip">768D XLM-R (frozen)</span>',
        '<span class="spec-chip">BiLSTM · 64 units</span>',
        '<span class="spec-chip">Residual Self-Attention</span>',
        '</div>',
        f'<div class="arch-terminal">{PROPOSED_ARCH_HTML}</div>',
        '</div>',
        '</details>'
    )

    md('<div class="sidebar-section-label" style="margin-top:1.25rem;">Benchmarks</div>')

    bench_rows_html = "".join(
        f'<tr><td style="text-align:left;">{name}</td><td>{acc}</td><td>{prec}</td><td>{rec}</td><td>{f1}</td></tr>'
        for name, acc, prec, rec, f1 in BENCHMARK_ROWS
    )

    md(
        '<details class="model-detail">',
        '<summary>Experimental Results</summary>',
        '<div class="detail-body">',
        '<table class="bench-table">',
        '<tr><th style="text-align:left;">Model</th><th>Acc</th><th>Prec</th><th>Rec</th><th>F1</th></tr>',
        bench_rows_html,
        '</table>',
        '</div>',
        '</details>'
    )

    md('<div class="sidebar-section-label" style="margin-top:1.25rem;">Classification</div>')

    md(
        '<p style="font-family:Space Mono,monospace;font-size:0.65rem;color:var(--text-secondary);line-height:1.7;">'
        '<span style="color:var(--error);">1</span>&nbsp; Clickbait<br>'
        '<span style="color:var(--success);">0</span>&nbsp; Non-clickbait</p>'
    )

    md(
        '<div style="font-family:Space Mono,monospace;font-size:0.55rem;color:var(--text-muted);line-height:1.7;margin-top:0.75rem;">'
        'XLM-R encoder<br>Frozen during training<br>Maximum sequence length: 128</div>'
    )


# =============================================================================
# MAIN CONTENT
# =============================================================================

if "processing" not in st.session_state:
    st.session_state.processing = False

_, center, _ = st.columns([1, 3, 1])

with center:

    # Step indicator is rendered exactly once per run, from session_state,
    # so it can never appear twice and is always in sync with the current
    # stage (input -> analyzing -> results).
    if st.session_state.get("results"):
        current_step = 3
    elif st.session_state.processing:
        current_step = 2
    else:
        current_step = 1

    render_steps(current_step)

    md('<div class="section-label">News Headline</div>')

    headline = st.text_area(
        "",
        placeholder="Enter a news headline to analyze...",
        height=130,
        label_visibility="collapsed",
        key="headline_input"
    )

    md('</div>')
    md("<br>")

    analyze = st.button("Analyze Headline", use_container_width=True)

    # -------------------------------------------------------------------
    # STAGE 1 -> 2: validate input, flip to "processing", rerun so the
    # step indicator immediately reflects "Analysis" before the (fast)
    # inference actually runs.
    # -------------------------------------------------------------------

    if analyze and not st.session_state.processing:
        if not models_loaded:
            st.error("The detection models could not be loaded.")
        elif not headline.strip():
            st.warning("Please enter a headline before running the analysis.")
        else:
            st.session_state.results = None
            st.session_state.pending_headline = headline
            st.session_state.processing = True
            st.rerun()

    # -------------------------------------------------------------------
    # STAGE 2: run both models, time them, store results, rerun to
    # STAGE 3 so everything lands in one clean, final render.
    # -------------------------------------------------------------------

    if st.session_state.processing:
        with st.spinner("Running both classification models..."):
            try:
                pending_headline = st.session_state.pending_headline

                t0 = time.perf_counter()
                baseline_prediction, baseline_probability = predict_baseline(
                    pending_headline, baseline_model, baseline_tokenizer, baseline_config
                )
                baseline_time = time.perf_counter() - t0

                t1 = time.perf_counter()
                proposed_prediction, proposed_probability = predict_proposed(
                    pending_headline, proposed_model, xlmr_tokenizer, xlmr_encoder, proposed_config
                )
                proposed_time = time.perf_counter() - t1

                st.session_state.results = {
                    "headline": pending_headline,
                    "baseline_prediction": baseline_prediction,
                    "baseline_probability": baseline_probability,
                    "baseline_time": baseline_time,
                    "proposed_prediction": proposed_prediction,
                    "proposed_probability": proposed_probability,
                    "proposed_time": proposed_time,
                }

            except Exception as e:
                st.error(f"Analysis failed: {e}")

            finally:
                st.session_state.processing = False

        st.rerun()

    # -------------------------------------------------------------------
    # STAGE 3: show results
    # -------------------------------------------------------------------

    if st.session_state.get("results"):
        results = st.session_state.results

        md("<br>")
        md('<div class="results-heading">Analysis Results</div>')
        md(
            '<div class="analyzed-headline">',
            '<div class="analyzed-label">Analyzed Headline</div>',
            f'<div class="analyzed-text">{results["headline"]}</div>',
            '</div>'
        )

        col1, col2 = st.columns(2, gap="large")

        with col1:
            render_model_card(
                "baseline",
                "BiLSTM",
                results["baseline_prediction"],
                results["baseline_probability"],
                results["baseline_time"],
                ["300D Trainable Embedding", "BiLSTM · 64 units", "No Attention"]
            )

        with col2:
            render_model_card(
                "proposed",
                "XLM-R + BiLSTM + Residual Self-Attention",
                results["proposed_prediction"],
                results["proposed_probability"],
                results["proposed_time"],
                ["768D XLM-R (frozen)", "BiLSTM · 64 units", "Residual Self-Attention"]
            )

        baseline_prediction = results["baseline_prediction"]
        proposed_prediction = results["proposed_prediction"]

        if baseline_prediction == proposed_prediction:
            if baseline_prediction == 1:
                comparison_text = "Both models classify the headline as clickbait."
            else:
                comparison_text = "Both models classify the headline as non-clickbait."
            comparison_color = "var(--accent)"
        else:
            comparison_text = "The models produce different classifications for this headline."
            comparison_color = "var(--warning)"

        if results["baseline_time"] > 0:
            speed_ratio = results["proposed_time"] / results["baseline_time"]
            speed_note = (
                f'XLM-R model ran {speed_ratio:.1f}× slower than the baseline '
                f'({results["proposed_time"] * 1000:.0f} ms vs {results["baseline_time"] * 1000:.0f} ms).'
            )
        else:
            speed_note = ""

        md(
            '<div class="comparison-panel">',
            '<div class="comparison-label">Model Comparison</div>',
            f'<div class="comparison-value" style="color:{comparison_color};">{comparison_text}</div>',
            f'<div class="comparison-value" style="margin-top:0.4rem;font-size:0.75rem;color:var(--text-muted);">{speed_note}</div>',
            '</div>'
        )


# =============================================================================
# FOOTER
# =============================================================================

md(
    '<div class="footer-text">XLM-RoBERTa + BiLSTM + Residual Self-Attention &nbsp;·&nbsp; Clickbait Detection Prototype</div>'
)