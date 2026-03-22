# scripts/08b_finetune_and_eval.py
"""
Fine-tunes FinBERT on your self-contained annotated_articles.csv with a train/val/test split,
early stopping (on val), class-weighted loss, and metadata tags (industry/scope/lang/time)
prefixed to the text. Evaluates once on the held-out test set and saves metrics/predictions.

Inputs (single CSV with text + labels):
- data/annotation/annotated_articles.csv
  Required headers:
    article_id,published_at,source,url,industry,language,title,description,text,y,scope,rationale

Outputs:
- models/finbert_finetuned/
- outputs/finetune_eval/test_results.json
- outputs/finetune_eval/predictions.csv
- outputs/finetune_eval/classification_report.txt

Run:
  python scripts/08b_finetune_and_eval.py
"""

from pathlib import Path
import json
import sys
import re
import random
import warnings
from dataclasses import fields

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ---- Project config (repo-local) ----
from _config import (
    ANNOTATED_GOLD_CSV,  # should point to data/annotation/annotated_articles.csv
    MARKETAUX_ARTICLES_CSV,
    OUTPUTS_DIR,
    MODELS_DIR,
    SEED,
)

# ---- Dependencies ----
try:
    import torch
    import transformers
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    try:
        from transformers import DataCollatorWithPadding, EarlyStoppingCallback
    except Exception:
        DataCollatorWithPadding = None
        EarlyStoppingCallback = None
except Exception as e:
    print("[ERROR] Missing deps:", e)
    print("Install with: pip install torch transformers accelerate scikit-learn pandas numpy")
    sys.exit(1)

# -------------------
# Settings (tunable)
# -------------------
MODEL_NAME = "ProsusAI/finbert"  # keep FinBERT as requested

OUTPUT_MODEL_DIR = MODELS_DIR / "finbert_finetuned"
OUTPUT_EVAL_DIR = OUTPUTS_DIR / "finetune_eval"

TEST_SIZE = 0.15      # final holdout
VAL_SIZE = 0.15       # validation (for early stopping / model selection)
RANDOM_STATE = SEED

# CPU-friendly defaults
NUM_EPOCHS = 8
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
EARLY_STOP_PATIENCE = 2
GRADIENT_ACCUMULATION_STEPS = 1  # bump on CPU if OOM

MAX_LENGTH = 192  # 160–256 is typical for headline+deck

# Data hygiene
KEEP_LANGS = {"en"}           # filter to languages you want to train on
BLACKLIST_SOURCES = set()     # e.g., {"zerohedge.com"} if you want to exclude
DROP_DUP_BY_URL = True
DROP_DUP_BY_TITLE_DESC = True

# Metadata tags in the input text (recommended ON)
USE_META_TAGS = True


# -------------------
# Repro & small utils
# -------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def _compose_text(row: pd.Series) -> str:
    """
    Build the training text from title/description/text with optional metadata tags.
    - Prefer 'text' if it's non-trivial; else fall back to "title — description".
    - Prefix tags like [IND_TECHNOLOGY] [SCOPE_MARKET] [LANG_EN] [Y2025] [Q1].
    """
    # Base content
    txt = _safe_str(row.get("text")).strip()
    if len(txt) < 20:
        title = _safe_str(row.get("title")).strip()
        desc = _safe_str(row.get("description")).strip()
        if title and desc:
            txt = f"{title} — {desc}"
        else:
            txt = title or desc

    # Collapse whitespace
    txt = re.sub(r"\s+", " ", txt).strip()

    if not USE_META_TAGS:
        return txt

    tags = []

    # Industry tag
    ind = _safe_str(row.get("industry")).strip()
    if ind:
        ind_tag = re.sub(r"[^A-Za-z0-9]+", "_", ind.upper())
        tags.append(f"[IND_{ind_tag}]")

    # Scope tag (normalized to COMPANY/SECTOR/MARKET if possible)
    scope = _safe_str(row.get("scope")).strip().lower()
    scope_map = {"company": "COMPANY", "sector": "SECTOR", "market": "MARKET"}
    if scope in scope_map:
        tags.append(f"[SCOPE_{scope_map[scope]}]")

    # Language tag (short)
    lang = _safe_str(row.get("language")).strip().upper()
    if lang and len(lang) <= 5:
        tags.append(f"[LANG_{lang}]")

    # Time tags (year + quarter)
    pub = _safe_str(row.get("published_at")).strip()
    if pub:
        try:
            dt = pd.to_datetime(pub, utc=True, errors="coerce")
            if pd.notna(dt):
                tags.append(f"[Y{dt.year}]")
                q = (dt.month - 1) // 3 + 1
                tags.append(f"[Q{q}]")
        except Exception:
            pass

    prefix = " ".join(tags).strip()
    return (prefix + " " + txt).strip() if prefix else txt


def _map_label(y):
    """Map y to {0,1,2}. Accepts ints or strings {negative,neutral,positive}."""
    if pd.isna(y):
        return None
    if isinstance(y, (int, np.integer)):
        y_int = int(y)
        if y_int in (0, 1, 2):
            return y_int
    # string labels
    y_str = str(y).strip().lower()
    str2id = {"negative": 0, "neg": 0, "0": 0,
              "neutral": 1, "neu": 1, "1": 1,
              "positive": 2, "pos": 2, "2": 2}
    return str2id.get(y_str, None)


# -------------------
# Data loading / cleaning
# -------------------
# Columns required from the thin gold CSV itself
REQUIRED_GOLD_COLS = ["article_id", "y"]
# Full set expected after merging with Marketaux
REQUIRED_COLS = [
    "article_id", "published_at", "source", "url",
    "industry", "language", "title", "description",
    "text", "y", "scope", "rationale"
]


def _merge_marketaux(df_gold: pd.DataFrame) -> pd.DataFrame:
    """Left-join Marketaux metadata onto gold annotations using article_id."""
    art_p = Path(MARKETAUX_ARTICLES_CSV)
    if not art_p.exists():
        print(f"[WARN] Marketaux articles CSV not found at {art_p}. Text/meta columns will be empty.")
        return df_gold
    df_art = pd.read_csv(art_p)
    keep = ["article_id", "published_at", "source", "url", "industry",
            "language", "title", "description", "text", "dominant_industry"]
    df_art = df_art[[c for c in keep if c in df_art.columns]].copy()
    df_gold["article_id"] = df_gold["article_id"].astype(str)
    df_art["article_id"]  = df_art["article_id"].astype(str)
    merged = df_gold.merge(df_art, on="article_id", how="left", suffixes=("", "_ma"))
    if "industry" not in merged.columns and "dominant_industry" in merged.columns:
        merged.rename(columns={"dominant_industry": "industry"}, inplace=True)
    if "text" not in merged.columns:
        merged["text"] = ""
    merged["text"] = merged["text"].fillna("")
    title = merged.get("title", pd.Series([""] * len(merged))).fillna("")
    desc  = merged.get("description", pd.Series([""] * len(merged))).fillna("")
    empty = merged["text"].str.len() == 0
    merged.loc[empty, "text"] = (title + " — " + desc).str.strip(" —")[empty]
    n_missing = int(merged["text"].eq("").sum())
    if n_missing > 0:
        print(f"[WARN] {n_missing} rows have empty text after Marketaux merge.")
    return merged


def _load_dataframe() -> pd.DataFrame:
    p = Path(ANNOTATED_GOLD_CSV)
    if not p.exists():
        print(f"[ERROR] Missing annotated CSV at {p}")
        return pd.DataFrame()

    df = pd.read_csv(p)

    # Validate thin gold file columns
    missing_gold = [c for c in REQUIRED_GOLD_COLS if c not in df.columns]
    if missing_gold:
        print(f"[ERROR] annotated_articles.csv missing required columns: {missing_gold}")
        return pd.DataFrame()

    # Merge text + metadata from Marketaux
    df = _merge_marketaux(df)

    # Fill any still-missing columns with empty strings (non-fatal)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = ""

    # Map labels
    df["labels"] = df["y"].apply(_map_label)
    df = df.dropna(subset=["labels"]).copy()
    df["labels"] = df["labels"].astype(int)

    # Language filter
    if KEEP_LANGS:
        df = df[df["language"].astype(str).str.lower().isin(KEEP_LANGS)].copy()

    # Source blacklist (optional)
    if BLACKLIST_SOURCES:
        df = df[~df["source"].astype(str).str.lower().isin({s.lower() for s in BLACKLIST_SOURCES})].copy()

    # Compose the final text
    df["text_final"] = df.apply(_compose_text, axis=1)
    df["text_final"] = df["text_final"].astype(str).str.strip()
    df = df[df["text_final"].str.len() > 10].copy()

    # De-dup
    if DROP_DUP_BY_URL and "url" in df.columns:
        df = df.drop_duplicates(subset=["url"]).copy()
    if DROP_DUP_BY_TITLE_DESC:
        td = (df["title"].fillna("") + " — " + df["description"].fillna("")).str.strip(" —")
        df = df.loc[~td.duplicated()].copy()

    # Keep only what we need for training
    out = df[["text_final", "labels"]].rename(columns={"text_final": "text"}).reset_index(drop=True)
    return out


# -------------------
# Dataset wrapper
# -------------------
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


# -------------------
# Metrics
# -------------------
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": macro_f1}


# -------------------
# Weighted Trainer (robust to API changes)
# -------------------
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        # Silence tokenizer deprecation warning in our context
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # HuggingFace may pass extra kwargs (e.g., num_items_in_batch); we ignore them
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# -------------------
# Version-robust TrainingArguments
# -------------------
def _build_training_args():
    arg_names = {f.name for f in fields(TrainingArguments)}

    base = dict(
        output_dir=str(OUTPUT_MODEL_DIR),
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS if "warmup_steps" in arg_names else None,
        weight_decay=WEIGHT_DECAY if "weight_decay" in arg_names else None,
        logging_steps=25,
        logging_dir=str(OUTPUT_EVAL_DIR / "logs"),
        seed=RANDOM_STATE,
        report_to="none" if "report_to" in arg_names else None,
        remove_unused_columns=False,  # safer with custom compute_loss/pop
    )
    # prune Nones
    base = {k: v for k, v in base.items() if v is not None}

    eval_key = "eval_strategy" if "eval_strategy" in arg_names else ("evaluation_strategy" if "evaluation_strategy" in arg_names else None)
    save_key = "save_strategy" if "save_strategy" in arg_names else None

    if eval_key:
        base[eval_key] = "epoch"
    if save_key:
        base[save_key] = "epoch"

    if "load_best_model_at_end" in arg_names and eval_key and save_key:
        base["load_best_model_at_end"] = True
        if "metric_for_best_model" in arg_names:
            base["metric_for_best_model"] = "macro_f1"
        if "greater_is_better" in arg_names:
            base["greater_is_better"] = True
        if "save_total_limit" in arg_names:
            base["save_total_limit"] = 1

    # Only keep supported keys
    base = {k: v for k, v in base.items() if k in arg_names}

    try:
        return TrainingArguments(**base)
    except ValueError:
        # fallback to steps if epoch strategies are rejected
        if eval_key:
            base[eval_key] = "steps"
        if save_key:
            base[save_key] = "steps"
        if "eval_steps" in arg_names:
            base["eval_steps"] = 200
        if "save_steps" in arg_names:
            base["save_steps"] = 200
        if "load_best_model_at_end" in arg_names:
            base["load_best_model_at_end"] = False
        return TrainingArguments(**base)


# -------------------
# Main
# -------------------
def main():
    set_all_seeds(RANDOM_STATE)

    # Env info
    print(f"[INFO] Python: {sys.version.split()[0]}")
    print(f"[INFO] torch: {torch.__version__}  | CUDA available: {torch.cuda.is_available()}")
    print(f"[INFO] transformers: {transformers.__version__}")

    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("[INFO] Loading annotated data…")
    df = _load_dataframe()
    if df.empty:
        print("[ERROR] No usable rows after loading/cleaning.")
        return

    print(f"[INFO] Loaded {len(df)} samples.")
    print("[INFO] Label distribution:", df["labels"].value_counts().sort_index().to_dict())

    # Train/Val/Test split (stratified)
    try:
        trval_df, te_df = train_test_split(
            df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["labels"]
        )
        val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
        tr_df, val_df = train_test_split(
            trval_df, test_size=val_ratio, random_state=RANDOM_STATE, stratify=trval_df["labels"]
        )
    except ValueError:
        # fallback without stratify
        trval_df, te_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
        tr_df, val_df = train_test_split(trval_df, test_size=val_ratio, random_state=RANDOM_STATE)

    print(f"[INFO] Split → train: {len(tr_df)}, val: {len(val_df)}, test: {len(te_df)}")

    train_texts = tr_df["text"].tolist()
    train_labels = tr_df["labels"].tolist()
    val_texts   = val_df["text"].tolist()
    val_labels  = val_df["labels"].tolist()
    test_texts  = te_df["text"].tolist()
    test_labels = te_df["labels"].tolist()

    # Tokenizer/Model
    print(f"[INFO] Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2},
    )

    # Tokenize
    train_enc = tokenizer(train_texts, truncation=True, padding=False, max_length=MAX_LENGTH)
    val_enc   = tokenizer(val_texts,   truncation=True, padding=False, max_length=MAX_LENGTH)
    test_enc  = tokenizer(test_texts,  truncation=True, padding=False, max_length=MAX_LENGTH)

    train_ds = SentimentDataset(train_enc, train_labels)
    val_ds   = SentimentDataset(val_enc,   val_labels)
    test_ds  = SentimentDataset(test_enc,  test_labels)

    data_collator = None
    if DataCollatorWithPadding is not None:
        try:
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        except Exception:
            data_collator = None

    # Class weights (compute on TRAIN only)
    classes_sorted = np.array(sorted(df["labels"].unique()))
    weights = compute_class_weight(class_weight="balanced", classes=classes_sorted, y=np.array(train_labels))
    # Ensure weight order is [0,1,2]
    class_weight_vector = np.zeros(3, dtype=np.float32)
    for c, w in zip(classes_sorted, weights):
        class_weight_vector[int(c)] = float(w)
    class_weights = torch.tensor(class_weight_vector, dtype=torch.float32)
    print(f"[INFO] Class weights: {class_weight_vector}")

    # Training args
    training_args = _build_training_args()

    # Trainer (evaluate on VAL; ES callback if available)
    # Note: transformers>=5.0 uses 'processing_class' instead of 'tokenizer'
    import transformers as _tv
    _tv_major = int(_tv.__version__.split(".")[0])
    _tok_kwarg = "processing_class" if _tv_major >= 5 else "tokenizer"

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        **{_tok_kwarg: tokenizer},
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    try:
        if EarlyStoppingCallback is not None:
            trainer = WeightedTrainer(
                **trainer_kwargs,
                class_weights=class_weights,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
            )
        else:
            trainer = WeightedTrainer(**trainer_kwargs, class_weights=class_weights)
    except TypeError:
        # older API may dislike compute_metrics in constructor
        trainer_kwargs.pop("compute_metrics", None)
        trainer = WeightedTrainer(**trainer_kwargs, class_weights=class_weights)

    # Train
    print("[INFO] Fine-tuning started… (this can be slow on CPU)")
    trainer.train()
    print("[INFO] Fine-tuning complete.")

    # Final evaluation on TEST set only
    print("[INFO] Evaluating on test set…")
    eval_res = trainer.evaluate(eval_dataset=test_ds)

    # Persist metrics
    results_path = OUTPUT_EVAL_DIR / "test_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(eval_res, f, indent=2)

    # Pretty print
    acc = eval_res.get("eval_accuracy")
    mf1 = eval_res.get("eval_macro_f1")
    print("\n" + "=" * 54)
    print("           FINAL RESULTS ON TEST SET")
    print("=" * 54)
    if acc is not None:
        print(f"  Accuracy : {acc:.4f}")
    if mf1 is not None:
        print(f"  Macro F1 : {mf1:.4f}")
    print("=" * 54)

    # Predictions + per-class report
    preds = trainer.predict(test_ds)
    yhat = np.argmax(preds.predictions, axis=1)

    report_txt = classification_report(test_labels, yhat, target_names=["negative", "neutral", "positive"])
    print("\nPer-class report:\n")
    print(report_txt)

    report_path = OUTPUT_EVAL_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    # Confusion matrix (printed only)
    cm = confusion_matrix(test_labels, yhat, labels=[0, 1, 2])
    print("Confusion matrix [rows=true, cols=pred]:")
    print(cm)

    # Save predictions
    pred_df = pd.DataFrame({
        "text": test_texts,
        "true_label": test_labels,
        "predicted_label": yhat
    })
    pred_out = OUTPUT_EVAL_DIR / "predictions.csv"
    pred_df.to_csv(pred_out, index=False, encoding="utf-8-sig")

    print(f"[OK] Metrics saved → {results_path}")
    print(f"[OK] Per-class report saved → {report_path}")
    print(f"[OK] Predictions saved → {pred_out}")

    # Save model + tokenizer
    try:
        trainer.save_model(str(OUTPUT_MODEL_DIR))
        tokenizer.save_pretrained(str(OUTPUT_MODEL_DIR))
        print(f"[OK] Fine-tuned model saved → {OUTPUT_MODEL_DIR}")
    except Exception as e:
        print(f"[WARN] Could not save model: {e}")


if __name__ == "__main__":
    main()
