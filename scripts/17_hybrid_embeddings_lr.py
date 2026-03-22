# scripts/17_hybrid_embeddings_lr.py
"""
Hybrid: FinBERT-CLS embeddings → Logistic Regression / optional EBM with SHAP

Datasets used if present:
  - FPB  (data/processed/fpb.csv)                    [expects columns: text,y]
  - FiQA (data/processed/fiqa_headlines.csv)         [expects columns: text,y]
  - GOLD (data/annotation/annotated_articles.csv)    [merged with Marketaux by article_id to get text]

Outputs per dataset:
  outputs/hybrid/<name>/metrics.json
  outputs/hybrid/<name>/confusion_matrix.csv
  outputs/hybrid/<name>/coef_top_dims.csv         (LR top/bottom dims by class)
  outputs/hybrid/<name>/shap_sample.csv           (LR SHAP top dims per sample)
  outputs/hybrid/<name>/ebm_metrics.json          (if EBM available)

Run: python scripts/17_hybrid_embeddings_lr.py
"""
from pathlib import Path
import json, re, math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from _config import (
    FPB_CSV, FIQA_HEADLINES_CSV, ANNOTATED_GOLD_CSV,
    MARKETAUX_ARTICLES_CSV, OUTPUTS_DIR, SEED
)

OUT_DIR = OUTPUTS_DIR / "hybrid"
MODEL_NAME = "ProsusAI/finbert"
MAX_LEN = 160
RANDOM_STATE = SEED

# ------------------------
# HF model + embeddings
# ------------------------
def _load_hf():
    try:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        mdl = AutoModel.from_pretrained(MODEL_NAME)
        mdl.eval()
        return tok, mdl
    except Exception as e:
        print(f"[ERROR] transformers load failed: {e}")
        return None, None

def _embed(texts, tok, mdl, batch_size=16):
    import torch
    device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
    mdl.to(device)
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, return_tensors="pt", truncation=True, max_length=MAX_LEN,
                      add_special_tokens=True, padding=True)
            for k in enc:
                enc[k] = enc[k].to(device)
            out = mdl(**enc)  # last_hidden_state
            cls = out.last_hidden_state[:, 0, :].detach().cpu().numpy()
            vecs.append(cls.astype(np.float32))
    return np.vstack(vecs)

# ------------------------
# Data loading
# ------------------------
def _load_dataset(name, path, text_col="text", y_col="y"):
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)

    if name.upper() == "GOLD":
        # Merge with Marketaux to bring in text/title/description
        if "text" not in df.columns:
            if "article_id" in df.columns and Path(MARKETAUX_ARTICLES_CSV).exists():
                m = pd.read_csv(MARKETAUX_ARTICLES_CSV)[["article_id", "text", "title", "description", "language"]]
                df = df.merge(m, on="article_id", how="left")
                # Compose text if needed
                if "text" not in df.columns:
                    parts = []
                    if "title" in df.columns: parts.append(df["title"].fillna(""))
                    if "description" in df.columns: parts.append(df["description"].fillna(""))
                    if parts:
                        df["text"] = pd.Series(parts).astype(str).agg(" — ".join, axis=0)
            else:
                print("[WARN] GOLD has no 'text' and cannot merge with Marketaux (missing article_id or Marketaux file). Skipping GOLD.")
                return None
        # Keep labeled only
        if y_col not in df.columns:
            print("[WARN] GOLD is missing 'y' column. Skipping.")
            return None
        df = df.dropna(subset=[y_col]).copy()

    # Basic cleaning
    if text_col not in df.columns or y_col not in df.columns:
        print(f"[WARN] {name}: columns '{text_col}' and/or '{y_col}' missing. Skipping.")
        return None
    df[text_col] = df[text_col].astype(str).fillna("")
    df = df[df[text_col].str.len() >= 10].copy()
    try:
        df[y_col] = df[y_col].astype(int)
    except Exception:
        # if labels are strings like {'negative','neutral','positive'}, map them
        mapping = {"negative":0, "neg":0, "neutral":1, "neu":1, "positive":2, "pos":2}
        df[y_col] = df[y_col].astype(str).str.lower().map(mapping)
        df = df.dropna(subset=[y_col]).copy()
        df[y_col] = df[y_col].astype(int)

    return df[[text_col, y_col]].rename(columns={text_col:"text", y_col:"y"})

def _safe_split(X, y, test_size=0.25, seed=RANDOM_STATE):
    """Stratify if feasible; else fall back to random split."""
    try:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=None)

# ------------------------
# Models + reports
# ------------------------
def _fit_lr(X, y):
    # Dense embeddings → standardize then multinomial LR (lbfgs)
    lr = Pipeline([
        ("sc", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(max_iter=400, solver="lbfgs", random_state=RANDOM_STATE))
    ])
    lr.fit(X, y)
    return lr

def _maybe_ebm(X_tr, y_tr, X_te, y_te):
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
    except Exception:
        return None, None
    ebm = ExplainableBoostingClassifier(random_state=RANDOM_STATE, interactions=0)
    ebm.fit(X_tr, y_tr)
    yhat = ebm.predict(X_te)
    metr = {"acc": float(accuracy_score(y_te, yhat)),
            "macro_f1": float(f1_score(y_te, yhat, average="macro"))}
    return ebm, metr

def _report(name, y_true, y_pred, out_dir):
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    # Save CM as CSV with nice labels
    cm_df = pd.DataFrame(cm, index=["true_0","true_1","true_2"], columns=["pred_0","pred_1","pred_2"])
    cm_df.to_csv(out_dir/"confusion_matrix.csv")
    (out_dir/"metrics.json").write_text(json.dumps(rep, indent=2))
    return rep

def _lr_feature_inspect(lr_pipeline, X_tr, X_te, out_dir, sample_max=64):
    """Top/bottom LR coefficients + SHAP on a small sample (top dims only)."""
    # Coefficients
    lr = lr_pipeline.named_steps["lr"]
    sc = lr_pipeline.named_steps["sc"]
    coef = lr.coef_  # shape (K, d)
    rows = []
    for cls, row in enumerate(coef):
        top_idx = np.argsort(row)[::-1][:20]
        bot_idx = np.argsort(row)[:20]
        for rank, i in enumerate(top_idx, 1):
            rows.append({"class": cls, "rank": rank, "dim": int(i), "coef": float(row[i]), "sign": "top"})
        for rank, i in enumerate(bot_idx, 1):
            rows.append({"class": cls, "rank": rank, "dim": int(i), "coef": float(row[i]), "sign": "bottom"})
    pd.DataFrame(rows).to_csv(out_dir/"coef_top_dims.csv", index=False)

    # SHAP (linear): background = scaled train slice; explain scaled test slice
    try:
        import shap
        bg = sc.transform(X_tr[:min(512, len(X_tr))])
        explainer = None
        try:
            explainer = shap.LinearExplainer(lr, bg)  # modern API
        except TypeError:
            explainer = shap.LinearExplainer(lr, bg, feature_dependence="independent")  # old kw fallback

        Xs = sc.transform(X_te[:min(sample_max, len(X_te))])
        shap_raw = explainer.shap_values(Xs)  # list[K] of (n, d)
        # Save only top 15 dims per sample per class to keep file small
        out_rows = []
        if isinstance(shap_raw, list):
            K = len(shap_raw)
            n, d = shap_raw[0].shape if K else (0, 0)
            for k in range(K):
                S = shap_raw[k]
                for i in range(S.shape[0]):
                    v = S[i]
                    top = np.argsort(np.abs(v))[::-1][:15]
                    for rank, j in enumerate(top, 1):
                        out_rows.append({
                            "sample_idx": int(i),
                            "class": int(k),
                            "rank": int(rank),
                            "dim": int(j),
                            "shap": float(v[j])
                        })
        pd.DataFrame(out_rows).to_csv(out_dir/"shap_sample.csv", index=False)
    except Exception as e:
        # SHAP is optional; if it fails, just move on
        print(f"[WARN] SHAP skipped: {e}")

# ------------------------
# Main
# ------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok, mdl = _load_hf()
    if tok is None:
        return

    datasets = []
    if Path(FPB_CSV).exists():                datasets.append(("FPB", FPB_CSV))
    if Path(FIQA_HEADLINES_CSV).exists():     datasets.append(("FiQA", FIQA_HEADLINES_CSV))
    if Path(ANNOTATED_GOLD_CSV).exists():     datasets.append(("GOLD", ANNOTATED_GOLD_CSV))

    if not datasets:
        print("[ERROR] No datasets found.")
        return

    for name, path in datasets:
        out = OUT_DIR / name.lower()
        out.mkdir(parents=True, exist_ok=True)
        df = _load_dataset(name, path)
        if df is None or df.empty:
            print(f"[WARN] Skipping {name}: empty or invalid.")
            continue

        texts = df["text"].astype(str).tolist()
        y = df["y"].astype(int).values

        print(f"[INFO] Embedding {name} (n={len(texts)}) …")
        X = _embed(texts, tok, mdl, batch_size=16)

        # Split (stratify if possible)
        X_tr, X_te, y_tr, y_te = _safe_split(X, y, test_size=0.25, seed=RANDOM_STATE)

        # LR
        lr = _fit_lr(X_tr, y_tr)
        yhat = lr.predict(X_te)
        _report(name, y_te, yhat, out)
        _lr_feature_inspect(lr, X_tr, X_te, out)

        # Optional EBM
        ebm, metr = _maybe_ebm(X_tr, y_tr, X_te, y_te)
        if ebm is not None:
            (out/"ebm_metrics.json").write_text(json.dumps(metr, indent=2))

        print(f"[DONE] Hybrid embeddings for {name} → {out}")

if __name__ == "__main__":
    main()
