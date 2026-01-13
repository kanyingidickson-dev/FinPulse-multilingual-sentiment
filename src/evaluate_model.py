import json
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/sentiment_model"
ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}

def evaluate(test_file, output_dir="reports"):
    logger.info("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except OSError:
        logger.error("Model not found. Please run train_model.py first.")
        return

    device = 0 if torch.cuda.is_available() else -1
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device, return_all_scores=True)

    logger.info("Loading test data...")
    texts = []
    true_labels = []
    languages = []
    
    with open(test_file, 'r') as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec['text'])
            true_labels.append(rec['label'])
            languages.append(rec.get('language', 'unknown'))

    logger.info(f"Running inference on {len(texts)} samples...")
    # Batch inference could be optimized, but sequential is fine for small/medium dataset scripts
    predictions = pipe(texts)
    
    # Process predictions to get max score label
    pred_labels = []
    for pred in predictions:
        # pred is a list of dicts [{'label': '..', 'score': ..}, ..]
        # We need to find the label with max score
        # Note: Model returns ID or label depending on config. 
        # Since we saved with id2label, pipeline usually returns labels like "positive"
        
        # Sort by score desc
        best = max(pred, key=lambda x: x['score'])
        pred_labels.append(best['label'])

    # --- Metrics Calculation ---
    
    print("\n=== Classification Report (Overall) ===")
    overall_report = classification_report(true_labels, pred_labels, output_dict=True)
    print(classification_report(true_labels, pred_labels))
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=["negative", "neutral", "positive"])
    
    # Language-wise breakdown
    df = pd.DataFrame({'true': true_labels, 'pred': pred_labels, 'lang': languages})
    
    lang_metrics = {}
    for lang in df['lang'].unique():
        subset = df[df['lang'] == lang]
        if len(subset) == 0: continue
        
        print(f"\n=== Classification Report ({lang.upper()}) ===")
        print(classification_report(subset['true'], subset['pred']))
        lang_metrics[lang] = classification_report(subset['true'], subset['pred'], output_dict=True)

    # Save metrics
    metrics_data = {
        "overall": overall_report,
        "confusion_matrix": cm.tolist(),
        "per_language": lang_metrics
    }
    
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)
        
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write(classification_report(true_labels, pred_labels))

    logger.info(f"Evaluation complete. Reports saved to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default="data/annotated_test.jsonl")
    args = parser.parse_args()
    
    evaluate(args.test_file)
