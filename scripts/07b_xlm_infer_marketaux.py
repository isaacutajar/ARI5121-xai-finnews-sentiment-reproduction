# scripts/07b_xlm_infer_marketaux.py
"""
Multilingual sentiment inference for non-EN Marketaux news.

Primary model: cardiffnlp/twitter-xlm-roberta-base-sentiment (3-way)
Fallback    : nlptown/bert-base-multilingual-uncased-sentiment (1–5★ → 3-way)

Reads:
  data/processed/marketaux/marketaux_news_articles.csv
Writes:
  outputs/multilingual/xlmr_marketaux_preds.csv
  outputs/multilingual/xlmr_lang_summary.csv
  outputs/multilingual/summary.json

Run: python scripts/07b_xlm_infer_marketaux.py
"""
from pathlib import Path
import json
import re
import pandas as pd
from _config import MARKETAUX_ARTICLES_CSV, OUTPUTS_DIR

OUT_DIR = OUTPUTS_DIR / "multilingual"

def _load_pipeline():
    """
    Try CardiffNLP XLM-R (SentencePiece, 3 classes). If it fails, try the
    nlptown 1–5★ model and map to 3 classes.
    Returns: (pipeline, mode) where mode in {"cardiff", "nlptown"}.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
        import torch
    except Exception as e:
        print(f"[ERROR] transformers/torch not installed: {e}")
        return None, None

    device = 0 if torch.cuda.is_available() else -1

    # --- Try CardiffNLP (preferred) ---
    try:
        tok = AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            use_fast=False  # robust on Windows; requires `pip install sentencepiece`
        )
        mdl = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        )
        pipe = TextClassificationPipeline(
            model=mdl, tokenizer=tok, device=device,
            return_all_scores=True, truncation=True, max_length=256
        )
        print("[OK] Loaded CardiffNLP XLM-R sentiment.")
        return pipe, "cardiff"
    except Exception as e:
        print(f"[WARN] CardiffNLP load failed: {e}")

    # --- Fallback: nlptown 1–5★ ---
    try:
        tok = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        mdl = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        pipe = TextClassificationPipeline(
            model=mdl, tokenizer=tok, device=device,
            return_all_scores=True, truncation=True, max_length=256
        )
        print("[OK] Loaded fallback: nlptown/bert-base-multilingual-uncased-sentiment.")
        return pipe, "nlptown"
    except Exception as e:
        print(f"[ERROR] Fallback model load failed: {e}")
        return None, None

def _predict(pipe, mode, texts):
    labels_3 = []
    probs_3 = []

    if mode == "cardiff":
        # Expect 3 dicts per item: negative / neutral / positive
        outs = pipe(texts)
        for o in outs:
            by = {d["label"].lower(): float(d["score"]) for d in o}
            # also handle 'label_0' style keys
            if not by:
                by = {}
                for d in o:
                    k = d["label"].lower()
                    if re.match(r"label_\d+", k):
                        # map by order if names are LABEL_0..2
                        # we assume pipeline order is [neg, neu, pos]
                        pass
                # fall back on index order
                p_neg, p_neu, p_pos = o[0]["score"], o[1]["score"], o[2]["score"]
            else:
                p_neg = by.get("negative", by.get("label_0", 0.0))
                p_neu = by.get("neutral",  by.get("label_1", 0.0))
                p_pos = by.get("positive", by.get("label_2", 0.0))

            probs_3.append((p_neg, p_neu, p_pos))
            if p_pos >= p_neu and p_pos >= p_neg:
                labels_3.append(2)
            elif p_neg >= p_neu and p_neg >= p_pos:
                labels_3.append(0)
            else:
                labels_3.append(1)
        return labels_3, probs_3

    elif mode == "nlptown":
        # Returns 5 dicts per item: '1 star'..'5 stars'
        outs = pipe(texts)
        for o in outs:
            by = {d["label"].lower(): float(d["score"]) for d in o}
            p1 = by.get("1 star", 0.0)
            p2 = by.get("2 stars", 0.0)
            p3 = by.get("3 stars", 0.0)
            p4 = by.get("4 stars", 0.0)
            p5 = by.get("5 stars", 0.0)
            # Map 1–2★ → neg, 3★ → neu, 4–5★ → pos
            p_neg = p1 + p2
            p_neu = p3
            p_pos = p4 + p5
            probs_3.append((p_neg, p_neu, p_pos))
            if p_pos >= p_neu and p_pos >= p_neg:
                labels_3.append(2)
            elif p_neg >= p_neu and p_neg >= p_pos:
                labels_3.append(0)
            else:
                labels_3.append(1)
        return labels_3, probs_3

    else:
        return None, None

def main():
    if not MARKETAUX_ARTICLES_CSV.exists():
        print(f"[ERROR] Missing: {MARKETAUX_ARTICLES_CSV}")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MARKETAUX_ARTICLES_CSV)
    df["text"] = df["text"].fillna("")
    df["language"] = df["language"].fillna("unk").str.lower()
    dnon = df[(df["language"] != "en") & (df["text"].str.len() >= 10)].copy()
    if dnon.empty:
        print("[INFO] No non-EN rows found.")
        return

    pipe, mode = _load_pipeline()
    if pipe is None:
        print("[WARN] Multilingual inference skipped (no model). Tip: pip install transformers sentencepiece torch")
        return

    # Batch for speed/memory
    B = 32
    all_labels, all_probs = [], []
    texts = dnon["text"].tolist()
    for i in range(0, len(texts), B):
        y, p = _predict(pipe, mode, texts[i:i+B])
        all_labels.extend(y)
        all_probs.extend(p)

    dnon["y_pred"] = all_labels
    dnon["prob_neg"] = [p[0] for p in all_probs]
    dnon["prob_neu"] = [p[1] for p in all_probs]
    dnon["prob_pos"] = [p[2] for p in all_probs]

    cols = ["article_id","published_at","source","url","language","dominant_industry","title","description","text",
            "y_pred","prob_neg","prob_neu","prob_pos"]
    dnon[cols].to_csv(OUT_DIR / "xlmr_marketaux_preds.csv", index=False, encoding="utf-8-sig")

    # per-language summary
    summ = (
        dnon.groupby(["language","y_pred"]).size().unstack(fill_value=0)
        .rename(columns={0:"neg",1:"neu",2:"pos"})
    )
    summ["total"] = summ.sum(axis=1)
    summ = summ.sort_values("total", ascending=False)
    summ.to_csv(OUT_DIR / "xlmr_lang_summary.csv")

    (OUT_DIR / "summary.json").write_text(json.dumps({
        "n_non_en": int(len(dnon)),
        "languages": {k: int(v) for k, v in dnon["language"].value_counts().to_dict().items()},
        "model_used": mode
    }, indent=2))

    print(f"[DONE] Multilingual inference ({mode}) → {OUT_DIR / 'xlmr_marketaux_preds.csv'}")

if __name__ == "__main__":
    main()
