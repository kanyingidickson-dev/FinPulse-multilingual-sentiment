import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from typing import List, Dict, Tuple
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text: str, lang: str = 'en') -> str:
    """
    Cleans text for sentiment analysis.
    
    Args:
        text: The raw text string.
        lang: Language code ('en' or 'sw').
        
    Returns:
        Cleaned text string.
    """
    # 1. Lowercase (standard for most transformer models unless case carries specific sentiment weight)
    # However, XLM-R is often cased, but standardizing helps with noise.
    # We will keep casing for XLM-R mostly, but doing a mild normalize here.
    # Actually, for deep learning models, aggressive cleaning (stopwords removal) is often discouraged.
    # We will focus on artifacts removal.
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove user mentions (e.g., @user) - privacy and usually irrelevant for sentiment logic in this context
    text = re.sub(r'@\w+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Specific handling could go here for Swahili if needed (e.g., specific punctuation or normalization)
    # For now, general cleaning works well for both in a transformer context.
    
    return text

def load_data(file_paths: List[str]) -> pd.DataFrame:
    """Loads and merges JSONL files into a DataFrame."""
    data = []
    for path in file_paths:
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    data.append(record)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in {path}")
                    
    return pd.DataFrame(data)

def preprocess_pipeline(
    input_files: List[str],
    train_output_path: str,
    test_output_path: str,
    test_size: float = 0.2,
    random_seed: int = 42
):
    """
    Main preprocessing pipeline: Load -> Clean -> Split -> Save.
    """
    logger.info("Starting preprocessing pipeline...")
    
    # 1. Load Data
    df = load_data(input_files)
    if df.empty:
        logger.error("No data loaded. Exiting.")
        return
    
    logger.info(f"Loaded {len(df)} records.")
    
    # 2. Clean Data
    logger.info("Cleaning text...")
    df['text_clean'] = df.apply(lambda row: clean_text(row['text'], row['language']), axis=1)
    
    # 3. Stratified Split
    # We want to maintain label distribution across train/test
    # We also want to ideally maintain language balance, but label is primary for classification.
    # Let's create a stratify column combining lang + label if possible, or just label.
    # For robustness, let's stratify by label.
    
    if 'label' not in df.columns:
        logger.error("Data missing 'label' column. Cannot split for training.")
        return

    logger.info(f"Splitting data (test_size={test_size}, seed={random_seed})...")
    
    # Handling class imbalance warnings if dataset is tiny
    stratify_col = df['label']
    if df['label'].value_counts().min() < 2:
        logger.warning("Some classes have fewer than 2 samples. Stratified split disabled.")
        stratify_col = None
        
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_col
    )
    
    # 4. Save
    logger.info(f"Saving train ({len(train_df)}) and test ({len(test_df)}) sets...")
    
    # We save the cleaned text as 'text' for the model input, or keep original?
    # Usually we train on the cleaned text. Let's overwrite 'text' or keep separate.
    # Transformers are robust, but let's use the cleaned version.
    
    output_columns = ['id', 'text_clean', 'language', 'label']
    
    train_df = train_df.rename(columns={'text_clean': 'text'}) # Use cleaned as main text
    test_df = test_df.rename(columns={'text_clean': 'text'})
    
    train_df[output_columns].to_json(train_output_path, orient='records', lines=True)
    test_df[output_columns].to_json(test_output_path, orient='records', lines=True)
    
    logger.info("Preprocessing complete.")

if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess data for sentiment analysis")
    parser.add_argument("--inputs", nargs='+', default=["data/social_en.jsonl", "data/news_sw.jsonl"], help="Input JSONL files")
    parser.add_argument("--train_out", default="data/annotated_train.jsonl", help="Output train file")
    parser.add_argument("--test_out", default="data/annotated_test.jsonl", help="Output test file")
    
    args = parser.parse_args()
    
    preprocess_pipeline(args.inputs, args.train_out, args.test_out)
