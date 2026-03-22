# scripts/04_lexicon_baselines.py
"""
Lexicon baselines on FPB + FiQA: LM wordlists (required), VADER (optional).

Reads:
  data/processed/fpb.csv, data/processed/fiqa_headlines.csv
  data/lexicons/lm/lm_{positive,negative,...}.txt

Writes:
  outputs/baselines/lexicon_metrics.csv
  outputs/baselines/preds_lexicon_{dataset}.csv
Run: python scripts/04_lexicon_baselines.py
"""
from pathlib import Path
import re, json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from _config import FPB_CSV, FIQA_HEADLINES_CSV, LEXICON_DIR, OUTPUTS_DIR, VADER_POS_THR, VADER_NEG_THR

OUT_DIR = OUTPUTS_DIR / "baselines"
TOKEN_RE = re.compile(r"[A-Za-z]+")

def _load_texts_y(path: Path):
    df = pd.read_csv(path)
    if "text" not in df.columns or "y" not in df.columns:
        raise ValueError(f"CSV must have columns text,y: {path}")
    df = df.dropna(subset=["text","y"]).copy()
    df["y"] = df["y"].astype(int)
    return df

def _load_lm_sets():
    pos = (LEXICON_DIR / "lm_positive.txt").read_text(encoding="utf-8").splitlines() if (LEXICON_DIR / "lm_positive.txt").exists() else []
    neg = (LEXICON_DIR / "lm_negative.txt").read_text(encoding="utf-8").splitlines() if (LEXICON_DIR / "lm_negative.txt").exists() else []
    return set(w.strip() for w in pos if w.strip()), set(w.strip() for w in neg if w.strip())

def _lm_score(text, pos_set, neg_set):
    toks = [t.lower() for t in TOKEN_RE.findall(text or "")]
    if not toks: 
        return 0.0, 0, 0
    p = sum(1 for t in toks if t in pos_set)
    n = sum(1 for t in toks if t in neg_set)
    return float(p - n), p, n

def _lm_label_from_score(score: float, pos_thr: float = 1.0, neg_thr: float = -1.0):
    if score >= pos_thr: return 2
    if score <= neg_thr: return 0
    return 1

def _maybe_vader_score(text: str):
    """Return compound score or None if VADER not available."""
    # Try vaderSentiment first (no external data)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS_SIA
        s = VS_SIA()
        return s.polarity_scores(text or "").get("compound", 0.0)
    except Exception:
        pass
    # Fall back to NLTK
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer as NLTK_SIA
        import nltk
        try: nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError: nltk.download("vader_lexicon")
        s = NLTK_SIA()
        return s.polarity_scores(text or "").get("compound", 0.0)
    except Exception:
        return None


def _vader_label(compound: float):
    if compound is None: return None
    if compound >= VADER_POS_THR: return 2
    if compound <= VADER_NEG_THR: return 0
    return 1

def _metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "f1_neg": f1_score(y_true, y_pred, average=None, labels=[0,1,2])[0],
        "f1_neu": f1_score(y_true, y_pred, average=None, labels=[0,1,2])[1],
        "f1_pos": f1_score(y_true, y_pred, average=None, labels=[0,1,2])[2],
    }

def _run_on_dataset(name: str, path: Path, pos_set, neg_set):
    df = _load_texts_y(path)
    # LM
    lm_scores = []
    lm_p = []
    lm_n = []
    for t in df["text"].tolist():
        s,p,n = _lm_score(t, pos_set, neg_set)
        lm_scores.append(s); lm_p.append(p); lm_n.append(n)
    df["lm_score"] = lm_scores
    df["y_lm"] = [ _lm_label_from_score(s) for s in lm_scores ]
    m_lm = _metrics(df["y"], df["y_lm"])

    # VADER (optional)
    vader_comps, y_vader = [], []
    use_vader = True
    for t in df["text"].tolist():
        c = _maybe_vader_score(t)
        if c is None:
            use_vader = False
            break
        vader_comps.append(c)
        y_vader.append(_vader_label(c))
    if use_vader:
        df["vader_compound"] = vader_comps
        df["y_vader"] = y_vader
        m_vader = _metrics(df["y"], df["y_vader"])
    else:
        m_vader = None

    # write preds
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out = df[["text","y","lm_score","y_lm"] + (["vader_compound","y_vader"] if m_vader else [])]
    df_out.to_csv(OUT_DIR / f"preds_lexicon_{name}.csv", index=False, encoding="utf-8-sig")

    return {"dataset": name, "model": "LM", **m_lm}, ({"dataset": name, "model": "VADER", **m_vader} if m_vader else None)

def main():
    pos_set, neg_set = _load_lm_sets()
    if not pos_set or not neg_set:
        print(f"[ERROR] Missing LM lists in {LEXICON_DIR}. Run 03_build_lm_lists.py first.")
        return

    rows = []
    for name, path in [("fpb", FPB_CSV), ("fiqa", FIQA_HEADLINES_CSV)]:
        if not path.exists():
            print(f"[WARN] Missing: {path}. Skipping {name}.")
            continue
        m_lm, m_vader = _run_on_dataset(name, path, pos_set, neg_set)
        rows.append(m_lm)
        if m_vader: rows.append(m_vader)

    if rows:
        pd.DataFrame(rows).to_csv(OUT_DIR / "lexicon_metrics.csv", index=False)
        print(f"[DONE] wrote metrics → {OUT_DIR / 'lexicon_metrics.csv'}")
    else:
        print("[WARN] nothing ran.")

if __name__ == "__main__":
    main()
