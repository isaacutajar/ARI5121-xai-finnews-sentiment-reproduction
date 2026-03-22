# scripts/08_eval_on_manual_annotations.py
"""
Evaluate on manual gold annotations.

Models:
  - LM lexicon (if lm_positive.txt / lm_negative.txt exist)
  - VADER (if installed)
  - LR(FPB), LR(FiQA)  (train on the fly if models missing)
  - FinBERT (if transformers model available)
  - WeakLR overall (if models/weaklr/lr_tfidf_marketaux_overall.joblib exists)

Reads:
  data/annotation/annotated_articles.csv   # columns: article_id,y,…
  data/processed/marketaux/marketaux_news_articles.csv (for text/meta merge)

Writes:
  outputs/gold_eval/global_metrics.csv
  outputs/gold_eval/per_industry_metrics.csv
  outputs/gold_eval/per_language_metrics.csv
  outputs/gold_eval/predictions_{MODEL}.csv
  outputs/gold_eval/confmats/{MODEL}_confmat_global.json

Run: python scripts/08_eval_on_manual_annotations.py
"""
from pathlib import Path
import re, json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

from _config import (
    ANNOTATED_GOLD_CSV, MARKETAUX_ARTICLES_CSV, OUTPUTS_DIR, MODELS_DIR,
    FPB_CSV, FIQA_HEADLINES_CSV, LEXICON_DIR,
    VADER_POS_THR, VADER_NEG_THR,  
)


OUT_DIR = OUTPUTS_DIR / "gold_eval"
CONF_DIR = OUT_DIR / "confmats"
TOKEN_RE = re.compile(r"[A-Za-z]+")

# ------------------------ helpers ------------------------

def _series_or_empty(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] with NaNs as '', or an empty series if col missing."""
    if col in df.columns:
        return df[col].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype=str)

def _ensure_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'text' column exists. If missing or partially NaN, compose from title/description.
    Safe even if title/description are absent.
    """
    df = df.copy()
    title = _series_or_empty(df, "title")
    desc  = _series_or_empty(df, "description")
    composed = (title + " — " + desc).str.strip(" —")
    if "text" not in df.columns:
        df["text"] = composed
    else:
        df["text"] = df["text"].fillna("")
        # if empty after fill, backfill from composed
        empty_mask = df["text"].str.len() == 0
        df.loc[empty_mask, "text"] = composed[empty_mask]
    return df

def _merge_from_marketaux(df_gold: pd.DataFrame) -> pd.DataFrame:
    """
    Merge text + metadata from Marketaux articles using article_id.
    """
    art_p = Path(MARKETAUX_ARTICLES_CSV)
    if not art_p.exists():
        print(f"[ERROR] Missing Marketaux articles for merge: {art_p}")
        return df_gold

    df_art = pd.read_csv(art_p)
    keep_cols = ["article_id","published_at","source","url","title","description","text",
                 "dominant_industry","language"]
    df_art = df_art[[c for c in keep_cols if c in df_art.columns]].copy()

    # Ensure types
    if "article_id" in df_gold.columns:
        df_gold["article_id"] = df_gold["article_id"].astype(str)
    if "article_id" in df_art.columns:
        df_art["article_id"] = df_art["article_id"].astype(str)

    merged = df_gold.merge(df_art, on="article_id", how="left", suffixes=("", "_ma"))

    # If 'industry' missing, create from dominant_industry
    if "industry" not in merged.columns and "dominant_industry" in merged.columns:
        merged.rename(columns={"dominant_industry": "industry"}, inplace=True)

    # Normalize language
    merged["language"] = _series_or_empty(merged, "language").str.lower().replace("", "unk")

    # Ensure text exists (after merge)
    merged = _ensure_text(merged)

    # Warn on any article_ids not found in Marketaux
    n_missing = int(merged["text"].eq("").sum())
    if n_missing > 0:
        print(f"[WARN] {n_missing} gold rows still have empty text after merge (missing in Marketaux?).")

    # Default industry
    if "industry" not in merged.columns:
        merged["industry"] = "Unknown"
    merged["industry"] = merged["industry"].fillna("Unknown").astype(str)

    return merged

def _load_gold() -> pd.DataFrame | None:
    p = Path(ANNOTATED_GOLD_CSV)
    if not p.exists():
        print(f"[ERROR] Missing: {p}")
        return None
    df = pd.read_csv(p)
    if "y" not in df.columns:
        print("[ERROR] Gold file must contain column 'y' in {0,1,2}.")
        return None

    # keep labeled only
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)

    if "article_id" not in df.columns:
        print("[ERROR] Gold file must contain 'article_id' to merge text/metadata.")
        return None

    # Merge text + meta from Marketaux
    df = _merge_from_marketaux(df)

    # Final sanitation
    df["text"] = df["text"].fillna("")
    df = df[df["text"].str.len() >= 1].copy()
    df["language"] = df["language"].fillna("unk").str.lower()

    return df

def _metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    per = f1_score(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_neg": float(per[0]),
        "f1_neu": float(per[1]),
        "f1_pos": float(per[2]),
        "n": int(len(y_true)),
    }, cm

# ------------------- model predictors -------------------

def _pred_lm(texts):
    pos_p = LEXICON_DIR / "lm_positive.txt"
    neg_p = LEXICON_DIR / "lm_negative.txt"
    if not (pos_p.exists() and neg_p.exists()):
        return None
    pos = set(w.strip().lower() for w in pos_p.read_text(encoding="utf-8").splitlines() if w.strip())
    neg = set(w.strip().lower() for w in neg_p.read_text(encoding="utf-8").splitlines() if w.strip())
    out = []
    for t in texts:
        toks = [x.lower() for x in TOKEN_RE.findall(t or "")]
        p = sum(1 for z in toks if z in pos)
        n = sum(1 for z in toks if z in neg)
        sc = p - n
        if sc >= 1: out.append(2)
        elif sc <= -1: out.append(0)
        else: out.append(1)
    return out

def _pred_vader(texts):
    """
    Robust VADER: try vaderSentiment (no external data), then fall back to NLTK.
    Uses thresholds from _config.py (VADER_POS_THR / VADER_NEG_THR).
    """
    sia = None
    # 1) Preferred: vaderSentiment (pip install vaderSentiment)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS_SIA
        sia = VS_SIA()
        backend = "vaderSentiment"
    except Exception:
        # 2) Fallback: NLTK (downloads resource if needed)
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer as NLTK_SIA
            import nltk
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon")
            sia = NLTK_SIA()
            backend = "nltk"
        except Exception as e:
            print(f"[WARN] VADER unavailable ({e}). Install with: pip install vaderSentiment")
            return None

    y = []
    for t in texts:
        c = sia.polarity_scores(t or "").get("compound", 0.0)
        if c >= VADER_POS_THR:
            y.append(2)
        elif c <= VADER_NEG_THR:
            y.append(0)
        else:
            y.append(1)
    print(f"[OK] VADER predictions using backend: {backend}")
    return y


def _get_or_train_lr(model_name: str, train_path: Path):
    mpath = MODELS_DIR / f"lr_tfidf_{model_name}.joblib"
    if mpath.exists():
        try:
            return joblib.load(mpath)
        except Exception:
            pass
    if not train_path.exists():
        return None
    df = pd.read_csv(train_path).dropna(subset=["text","y"]).copy()
    df["y"] = df["y"].astype(int)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=2)),
        ("lr", LogisticRegression(max_iter=200, C=2.0, solver="lbfgs")),
    ])
    pipe.fit(df["text"].tolist(), df["y"].values)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, mpath)
    return pipe

def _pred_lr(pipe, texts):
    if pipe is None: return None
    try:
        return pipe.predict(texts).tolist()
    except Exception:
        return None

def _pred_finbert(texts):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
    except Exception:
        return None
    try:
        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, return_all_scores=False, truncation=True)
    except Exception:
        return None
    outs = pipe(texts, batch_size=32)
    def map_lbl(lbl):
        s = (lbl or "").lower()
        if "pos" in s: return 2
        if "neg" in s: return 0
        return 1
    return [map_lbl(o["label"]) for o in outs]

def _pred_weaklr_overall(texts):
    p = MODELS_DIR / "weaklr" / "lr_tfidf_marketaux_overall.joblib"
    if not p.exists(): return None
    try:
        pipe = joblib.load(p)
        return pipe.predict(texts).tolist()
    except Exception:
        return None

# ------------------- evaluation writers -------------------

def _eval_and_write(name: str, y_true, y_pred, df_gold):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CONF_DIR.mkdir(parents=True, exist_ok=True)
    if y_pred is None:
        print(f"[INFO] Skipping {name}: predictions unavailable.")
        return None
    metr, cm = _metrics(y_true, y_pred)
    # Confusion matrix
    (CONF_DIR / f"{name}_confmat_global.json").write_text(
        json.dumps({"labels":[0,1,2],"matrix":cm.tolist()}, indent=2)
    )
    # Predictions table
    pred_df = pd.DataFrame({
        "article_id": df_gold.get("article_id", pd.Series([None]*len(y_true))),
        "industry": df_gold.get("industry", pd.Series(["Unknown"]*len(y_true))),
        "language": df_gold.get("language", pd.Series(["unk"]*len(y_true))),
        "text": df_gold["text"],
        "y_true": y_true,
        "y_pred": y_pred
    })
    pred_df.to_csv(OUT_DIR / f"predictions_{name}.csv", index=False, encoding="utf-8-sig")
    return {"model": name, **metr}

# --------------------------- main ---------------------------

def main():
    df = _load_gold()
    if df is None or df.empty:
        return

    # Base arrays
    y_true = df["y"].astype(int).tolist()
    texts  = df["text"].tolist()

    results = []

    # LM
    y = _pred_lm(texts)
    results.append(_eval_and_write("LM", y_true, y, df))

    # VADER
    y = _pred_vader(texts)
    results.append(_eval_and_write("VADER", y_true, y, df))

    # LR FPB
    lr_fpb = _get_or_train_lr("fpb", Path(FPB_CSV))
    y = _pred_lr(lr_fpb, texts)
    results.append(_eval_and_write("LR_FPB", y_true, y, df))

    # LR FiQA
    lr_fiqa = _get_or_train_lr("fiqa", Path(FIQA_HEADLINES_CSV))
    y = _pred_lr(lr_fiqa, texts)
    results.append(_eval_and_write("LR_FIQA", y_true, y, df))

    # FinBERT
    y = _pred_finbert(texts)
    results.append(_eval_and_write("FINBERT", y_true, y, df))

    # WeakLR overall
    y = _pred_weaklr_overall(texts)
    results.append(_eval_and_write("WEAKLR_OVERALL", y_true, y, df))

    results = [r for r in results if r is not None]
    if not results:
        print("[WARN] No models produced predictions. Check dependencies.")
        return

    # Global metrics
    pd.DataFrame(results).to_csv(OUT_DIR / "global_metrics.csv", index=False)

    # --- Per-industry metrics (robust: join on article_id) ---
    rows = []
    for ind, dgrp in df.groupby("industry"):
        ids = set(dgrp["article_id"].astype(str))
        yt = dgrp["y"].tolist()
        for r in results:
            name = r["model"]
            dp = pd.read_csv(OUT_DIR / f"predictions_{name}.csv")
            dp_ids = dp["article_id"].astype(str).isin(ids)
            y_hat = dp.loc[dp_ids, "y_pred"].tolist()
            if len(y_hat) != len(yt):  # if any mismatch, re-align by merge
                merged = dgrp[["article_id","y"]].merge(
                    dp[["article_id","y_pred"]], on="article_id", how="inner"
                )
                yt_m = merged["y"].tolist()
                y_hat = merged["y_pred"].tolist()
                if not yt_m:
                    continue
                m, _ = _metrics(yt_m, y_hat)
            else:
                m, _ = _metrics(yt, y_hat)
            rows.append({"model": name, "industry": ind, **m})
    if rows:
        pd.DataFrame(rows).to_csv(OUT_DIR / "per_industry_metrics.csv", index=False)

    # --- Per-language metrics (robust: join on article_id) ---
    rows = []
    for lang, dgrp in df.groupby("language"):
        ids = set(dgrp["article_id"].astype(str))
        yt = dgrp["y"].tolist()
        for r in results:
            name = r["model"]
            dp = pd.read_csv(OUT_DIR / f"predictions_{name}.csv")
            dp_ids = dp["article_id"].astype(str).isin(ids)
            y_hat = dp.loc[dp_ids, "y_pred"].tolist()
            if len(y_hat) != len(yt):
                merged = dgrp[["article_id","y"]].merge(
                    dp[["article_id","y_pred"]], on="article_id", how="inner"
                )
                yt_m = merged["y"].tolist()
                y_hat = merged["y_pred"].tolist()
                if not yt_m:
                    continue
                m, _ = _metrics(yt_m, y_hat)
            else:
                m, _ = _metrics(yt, y_hat)
            rows.append({"model": name, "language": lang, **m})
    if rows:
        pd.DataFrame(rows).to_csv(OUT_DIR / "per_language_metrics.csv", index=False)

    print(f"[DONE] Wrote metrics → {OUT_DIR}")

if __name__ == "__main__":
    main()
