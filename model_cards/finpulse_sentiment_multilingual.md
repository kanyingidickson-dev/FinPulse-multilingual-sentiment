# FinPulse Multilingual Sentiment Model Card

## Model Overview
This model is a **multilingual sentiment classifier** fine-tuned on
English and Swahili financial and social text.

**Primary use case:**
- Financial news sentiment analysis
- Social sentiment monitoring
- Regional market mood tracking (East Africa focus)

**Labels:**
- Positive
- Neutral
- Negative

---

## Model Architecture
- Base model: `xlm-roberta-base`
- Fine-tuned for 3-class sentiment classification
- Tokenization: SentencePiece (subword-based)

---

## Training Data
- English: Social media and public financial commentary
- Swahili: Financial news and public discourse
- Format: JSONL (`text`, `language`, `label`)
- Data size: Small demo-scale (illustrative only)

⚠️ **Note:** Datasets are for demonstration and are not representative of live markets.

---

## Evaluation Metrics
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Language-wise breakdown (EN vs SW)

---

## Intended Use
✅ Market sentiment exploration  
✅ NLP research & evaluation  
✅ FinTech prototyping  

❌ Not for automated trading  
❌ Not for financial advice  
❌ Not for real-time risk decisions  

---

## Ethical Considerations
- Sentiment may reflect media or social bias
- Swahili financial language varies by region
- Predictions should always be reviewed by humans

---

## Limitations
- Limited training data
- No real-time updates
- Does not capture sarcasm reliably
- Financial terminology drift over time

---

## License
Apache License 2.0
