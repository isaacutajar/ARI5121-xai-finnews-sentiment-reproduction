# scripts/_config.py
"""
Centralized constants for the pipeline (hardcoded, zero-arg scripts import this).

Edit here if paths/thresholds/targets change.
"""

from pathlib import Path

# --------- PATHS ---------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_MARKETAUX_JSON = RAW_DIR / "marketaux" / "news.json"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_MARKETAUX_DIR = PROCESSED_DIR / "marketaux"

ANNOTATION_DIR = DATA_DIR / "annotation"
ANNOTATION_BATCHES_DIR = ANNOTATION_DIR / "batches"

LEXICON_DIR = DATA_DIR / "lexicons" / "lm"

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# --------- FIXED INDUSTRY SET (Reviewer request: all sectors) ---------
INDUSTRIES = [
    "Communication Services",
    "Consumer Cyclical",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Services",
    "Utilities",
    "Technology",
]

# --------- GENERAL SETTINGS ---------
SEED = 42

# Weak-label thresholds (VADER-like) used elsewhere for consistency
VADER_NEG_THR = -0.05
VADER_POS_THR =  0.05

# Marketaux ingestion options
DROP_INDICES_ETFS_FX = True   # heuristics to exclude index/ETF/FX tickers from pairs table
KEEP_DUP_URL_FIRST = True      # de-dup by URL in articles table

# Language handling (Reviewer: include non-English)
TARGET_LANGS = ["en", "es", "de", "fr", "it", "pt", "nl", "sv", "fi", "no", "da", "ro"]
NON_EN_MIN_SHARE = 0.20        # at least 20% non-English in annotation batches

# Annotation batches
ANNOTATION_TARGET_TOTAL = 1500  
ANNOTATION_BATCH_SIZE = 200    # Writes batch_01.csv, batch_02.csv, ...
TEXT_COLUMNS_FOR_ANNOT = ["title", "description"]

# File names (derived)
FPB_CSV = PROCESSED_DIR / "fpb.csv"
FIQA_HEADLINES_CSV = PROCESSED_DIR / "fiqa_headlines.csv"

MARKETAUX_ARTICLES_CSV = PROCESSED_MARKETAUX_DIR / "marketaux_news_articles.csv"
MARKETAUX_PAIRS_CSV    = PROCESSED_MARKETAUX_DIR / "marketaux_news_pairs.csv"

ANNOTATED_GOLD_CSV = ANNOTATION_DIR / "annotated_articles.csv"  # your existing gold file (for leakage guard)

# Small helpers
def ensure_dirs():
    for p in [
        DATA_DIR, RAW_DIR, PROCESSED_DIR, PROCESSED_MARKETAUX_DIR,
        ANNOTATION_DIR, ANNOTATION_BATCHES_DIR, LEXICON_DIR,
        MODELS_DIR, OUTPUTS_DIR
    ]:
        p.mkdir(parents=True, exist_ok=True)
