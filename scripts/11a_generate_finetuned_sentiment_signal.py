# scripts/11a_generate_finetuned_sentiment_signal.py
"""
Generates a daily sentiment signal using the fine-tuned FinBERT model.

This script loads the best-performing model, runs inference on all English-language
articles from the Marketaux corpus, and computes a daily mean sentiment score
(-1 for negative, 0 for neutral, +1 for positive) for each industry sector.

Reads:
- models/finbert_finetuned/
- data/processed/marketaux/marketaux_news_articles.csv

Writes:
- outputs/finetune_eval/daily_sentiment_finetuned.csv
"""
from pathlib import Path
import pandas as pd
import torch
import re
import warnings

# Suppress Hugging Face UserWarnings about pipeline creation
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.pipelines.text_classification')

# --- Configuration ---
# Ensure paths are relative to the project root, regardless of where the script is run
try:
    from _config import MODELS_DIR, MARKETAUX_ARTICLES_CSV, OUTPUTS_DIR, SEED
except ImportError:
    # Fallback for direct execution or different project structure
    # This assumes the script is in a 'scripts' directory one level below the project root
    ROOT = Path(__file__).resolve().parents[1]
    MODELS_DIR = ROOT / "models"
    MARKETAUX_ARTICLES_CSV = ROOT / "data" / "processed" / "marketaux" / "marketaux_news_articles.csv"
    OUTPUTS_DIR = ROOT / "outputs"
    SEED = 42
    print("Warning: Could not import from _config.py, using fallback paths.")


MODEL_PATH = MODELS_DIR / "finbert_finetuned"
OUTPUT_DIR = OUTPUTS_DIR / "finetune_eval"
OUTPUT_FILE = OUTPUT_DIR / "daily_sentiment_finetuned.csv"

BATCH_SIZE = 32  # Adjust based on your GPU/CPU memory

# --- Main Logic ---

def _load_pipeline():
    """Loads the fine-tuned text classification pipeline."""
    if not MODEL_PATH.exists() or not (MODEL_PATH / "config.json").exists():
        print(f"[ERROR] Fine-tuned model not found at: {MODEL_PATH}")
        print("Please run script '08b_finetune_and_eval.py' first.")
        return None, None

    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TextClassificationPipeline
        )
    except ImportError:
        print("[ERROR] `transformers` library is not installed. Please install it: pip install transformers torch")
        return None, None

    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'cuda:0' if device == 0 else 'cpu'}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    # The pipeline handles tokenization, batching, and moving data to the device
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=False,
        truncation=True
    )

    # Dynamically get the label mapping from the model's config
    # FinBERT labels: {'negative': 0, 'neutral': 1, 'positive': 2}
    label2id = model.config.label2id
    sentiment_map = {label: (id-1 if label=='neutral' else (1 if label=='positive' else -1)) for label, id in label2id.items()}
    # Fallback if map is weirdly ordered or missing
    if 'positive' not in sentiment_map: sentiment_map['positive'] = 1
    if 'negative' not in sentiment_map: sentiment_map['negative'] = -1
    if 'neutral' not in sentiment_map: sentiment_map['neutral'] = 0


    print(f"Loaded fine-tuned model from: {MODEL_PATH}")
    print(f"Sentiment mapping created: {sentiment_map}")
    return pipe, sentiment_map


def main():
    """Main execution function."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pipeline, sentiment_map = _load_pipeline()
    if pipeline is None:
        return

    # Load and filter articles
    if not MARKETAUX_ARTICLES_CSV.exists():
        print(f"[ERROR] Marketaux articles file not found at: {MARKETAUX_ARTICLES_CSV}")
        return

    print("Loading and preparing articles...")
    df = pd.read_csv(MARKETAUX_ARTICLES_CSV, parse_dates=["published_at"])
    df["text"] = df["text"].fillna("").astype(str)
    df["language"] = df["language"].fillna("unk").str.lower()
    df["dominant_industry"] = df["dominant_industry"].fillna("Unknown").replace({"": "Unknown"})
    
    # Filter for English articles with sufficient text
    df_en = df[df["language"].eq("en") & (df["text"].str.len() >= 20)].copy()
    if df_en.empty:
        print("[ERROR] No eligible English-language articles found.")
        return

    texts_to_process = df_en["text"].tolist()
    print(f"Found {len(texts_to_process)} English articles to process...")

    # Run inference in batches
    all_preds = []
    print(f"Running inference with batch size {BATCH_SIZE}...")
    # The pipeline is highly optimized for batch processing
    results = pipeline(texts_to_process, batch_size=BATCH_SIZE)
    
    for result in results:
        label = result['label']
        score = sentiment_map.get(label, 0) # Default to neutral if label not in map
        all_preds.append(score)

    df_en["sentiment_score"] = all_preds
    print("Inference complete.")

    # Aggregate to daily sentiment per industry
    df_en["date"] = df_en["published_at"].dt.date
    daily_signal = (
        df_en.groupby(["date", "dominant_industry"])["sentiment_score"]
        .mean()
        .reset_index()
        .rename(columns={"dominant_industry": "industry", "sentiment_score": "sent_mean"})
    )

    # Sort and save
    daily_signal = daily_signal.sort_values(["industry", "date"])
    daily_signal.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[SUCCESS] Daily sentiment signal from fine-tuned model saved to:")
    print(f"-> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()