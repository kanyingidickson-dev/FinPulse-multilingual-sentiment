import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import logging
import json
import sys

# Configure logging
logging.basicConfig(level=logging.ERROR) # Only errors to keep output clean for pipe usage

MODEL_PATH = "models/sentiment_model"

class SentimentPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, device=self.device, return_all_scores=True)

    def predict(self, text, lang="unknown"):
        # Run inference
        out = self.pipe(text)
        # out is [[{'label': '..', 'score': ..}, ...]] (if list inputs) or single list if single input
        # Pipeline behavior varies slightly by version, but usually list of lists for list input.
        
        # handle single input list wrapper
        if isinstance(out[0], list):
            preds = out[0]
        else:
            preds = out
            
        # Get best class
        best = max(preds, key=lambda x: x['score'])
        
        result = {
            "text": text,
            # "language": lang, # logic to detect lang could be here, but we pass it through
            "predicted_label": best['label'],
            "confidence": round(best['score'], 4),
            "all_scores": {p['label']: round(p['score'], 4) for p in preds}
        }
        return result

if __name__ == "__main__":
    # Simple CLI usage
    if len(sys.argv) > 1:
        text_input = sys.argv[1]
        
        try:
            predictor = SentimentPredictor()
            res = predictor.predict(text_input)
            print(json.dumps(res, indent=2))
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Usage: python src/predict.py 'Your text string here'")
