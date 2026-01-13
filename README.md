# FinPulse — Multilingual Financial Sentiment Intelligence

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![Status](https://img.shields.io/badge/Status-mvp-orange)

**Multilingual financial sentiment analysis pipeline (English & Swahili) using transformer-based models. Designed for FinTech use cases including market sentiment tracking, risk signals, and regional financial intelligence.**

## 1. Project Overview

### FinPulse: The Mission
FinPulse is a multilingual sentiment intelligence system designed for financial news and social discourse analysis. It leverages transformer-based NLP models to extract sentiment signals from English and Swahili text, supporting FinTech use cases such as market sentiment tracking, risk monitoring, and regional economic analysis.

### Problem Statement
Financial sentiment analysis is traditionally dominated by English models. However, emerging markets in East Africa rely heavily on Swahili for local financial news, mobile money (M-Pesa) discourse, and regional trade.

This project builds a **Unified Multilingual Sentiment Model** capable of understanding market mood in both languages, handling the specific nuances of financial terminology and social noise.

### FinTech Use Case
Potential applications include:
- **Market Sentiment Tracking**: Gauge bullish/bearish trends from news headlines.
- **Risk Monitoring**: Detect negative sentiment spikes in social chatter regarding banking apps or regulations.
- **Regional Analysis**: Unlock insights from Swahili-speaking markets often ignored by global models.

## 2. Dataset Description

The system processes two primary data streams (simulated for this repo):

1.  **Financial News (Swahili)**: Formal language, economic terminology.
2.  **Social Media (English)**: Informal, noisy, user-generated content regarding banking/fintech.

**Labels**:
- `positive`: Bullish signaling, praise, growth, profit.
- `negative`: Bearish signaling, complaints, loss, inflation fears.
- `neutral`: Factual reporting, questions, statements without sentiment.

**Preprocessing Strategy**:
- Multilingual cleaning (URL removal, user anonymization).
- Language-aware tokenization using SentencePiece (via XLM-RoBERTa).
- Stratified splitting to maintain class balance.

## 3. Model Architecture

We use **XLM-RoBERTa (base)**, a powerful cross-lingual transformer model.

**Why XLM-R?**
- **Multilingual Mastery**: Pretrained on 100+ languages, showing superior performance on low-resource languages like Swahili compared to standard BERT.
- **Contextual Understanding**: Handles code-switching (mixing EN/SW) often found in regional tech discourse.
- **Transfer Learning**: Lessons learned from English financial data partially transfer to Swahili tasks.

**Governance**: 
A detailed [Model Card](model_cards/finpulse_sentiment_multilingual.md) is included, outlining ethical considerations, limitations, and intended use.

## 4. Engineering Standards

- **Reproducibility**: Fixed seeds for splitting and training.
- **Modularity**: Clean separation of concerns (`src/preprocess`, `src/train`, `src/evaluate`).
- **Data Governance**: Human-in-the-loop annotation tool included (`src/annotate.py`).

## 5. How to Run

### Installation
```bash
pip install -r requirements.txt
```

### 1. Preprocess Data
Clean and split raw JSONL data into training and test sets.
```bash
python src/preprocess.py
```

### 2. Train Model
Fine-tune XLM-RoBERTa on the annotated data.
```bash
python src/train_model.py
```
*(This will save the model to `models/sentiment_model`)*

### 3. Evaluate
Generate classification reports and confusion matrices.
```bash
python src/evaluate_model.py
```

### 4. Inference
Predict sentiment on new text.
```bash
python src/predict.py "Inflation is killing small businesses."
```

## 6. Sample Predictions

### English
> **Input**: "Fees on international transfers are getting ridiculous."  
> **Prediction**: `negative` (Confidence: 0.98)

> **Input**: "New fintech regulations are expected to boost digital payments adoption."  
> **Prediction**: `positive` (Confidence: 0.92)

### Swahili
> **Input**: "Uchumi unazidi kuimarika kutokana na biashara ya kimataifa." (The economy is strengthening due to international trade.)  
> **Prediction**: `positive` (Confidence: 0.95)

> **Input**: "Bei ya mafuta imepanda tena." (Fuel prices have gone up again.)  
> **Prediction**: `negative` (Confidence: 0.96)

## License

Licensed under the **Apache License 2.0**.  
See [LICENSE](LICENSE) for details.

---
*Disclaimer: This project contains sample data and models for demonstration purposes. It is not financial advice and should not be used for automated trading without extensive validation.*
