import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "xlm-roberta-base" # Strong multilingual support
MAX_LENGTH = 128
OUTPUT_DIR = "models/sentiment_model"
NUM_LABELS = 3
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}

def compute_metrics(eval_pred):
    """Compute metrics for HuggingFace Trainer."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': acc,
        'f1_weighted': f1
    }

def load_dataset(file_path):
    """Load JSONL into HuggingFace Dataset."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    # Map string labels to integers
    df['label'] = df['label'].map(LABEL_MAP)
    # Remove records with invalid labels
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    return Dataset.from_pandas(df)

def train(train_file, test_file):
    
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    logger.info("Loading datasets...")
    train_dataset = load_dataset(train_file)
    eval_dataset = load_dataset(test_file)
    
    logger.info("Tokenizing datasets...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    eval_tokenized = eval_dataset.map(tokenize_function, batched=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP
    )
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8, # Small batch for demo/CPU compatibility
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_dir='./logs',
        logging_steps=10,
        seed=42,
        report_to="none" # Disable wandb/mlflow for this local demo
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Saving best model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/annotated_train.jsonl")
    parser.add_argument("--test_file", default="data/annotated_test.jsonl")
    args = parser.parse_args()
    
    train(args.train_file, args.test_file)
