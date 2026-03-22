# scripts/10_weak_label_bias_audit.py
"""
Weak-label bias audit: GOLD vs WEAK (VADER→LM) on the gold corpus.

Procedure:
  1) Load gold annotations (text, y, industry/language if available).
  2) On the TRAIN split only, generate WEAK labels with VADER (LM fallback).
  3) Fit two LR+TFIDF models with a **shared vocabulary**:
       - GOLD model: train on (X_train, y_gold)
       - WEAK model: train on (X_train, y_weak)
  4) Evaluate both on the held-out TEST gold set; record acc/mF1 and deltas.
  5) SHAP explanation divergence on TEST:
       - per-class top-100 tokens → Spearman rank corr, Jaccard, JSD.
  6) Faithfulness (deletion): remove top-k tokens by per-sample SHAP magnitude,
     re-predict; compute flip-rate and avg prob drop curves for k∈{1,3,5,10}.

Reads:
  data/annotation/annotated_articles.csv
  data/processed/marketaux/marketaux_news_articles.csv  (for text join)
Writes:
  outputs/bias_audit/metrics_overall.json
  outputs/bias_audit/shap_rank_corr.csv
  outputs/bias_audit/flip_curves_gold.csv
  outputs/bias_audit/flip_curves_weak.csv
  outputs/bias_audit/token_flip_curves.png
  outputs/bias_audit/by_industry/{slug}_metrics.json   (if industry present)
Run: python scripts/10_weak_label_bias_audit.py
"""
from pathlib import Path
import json, re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

from _config import (
    ANNOTATED_GOLD_CSV, MARKETAUX_ARTICLES_CSV, OUTPUTS_DIR, SEED
)

OUT_DIR = OUTPUTS_DIR / "bias_audit"
IND_DIR = OUT_DIR / "by_industry"
RNG = np.random.default_rng(SEED)

def _slug(s): 
    return re.sub(r"[^a-z0-9]+","-", (s or "unknown").lower()).strip("-")

# ----------------------------
# Data loading / weak labeling
# ----------------------------
def _load_gold():
    p = Path(ANNOTATED_GOLD_CSV)
    if not p.exists():
        print(f"[ERROR] Missing gold file: {p}")
        return None

    gold = pd.read_csv(p)

    if "y" not in gold.columns:
        print("[ERROR] Gold file must contain 'y' in {0,1,2} or string labels.")
        return None

    if "article_id" not in gold.columns:
        print("[ERROR] Gold file must contain 'article_id' to join text from MarketAux.")
        return None

    ma_p = Path(MARKETAUX_ARTICLES_CSV)
    if not ma_p.exists():
        print(f"[ERROR] Missing MarketAux articles CSV: {ma_p} (needed to get text).")
        return None

    ma = pd.read_csv(ma_p)

    if "article_id" not in ma.columns:
        if "id" in ma.columns:
            ma = ma.rename(columns={"id": "article_id"})
        else:
            print("[ERROR] MarketAux file has neither 'article_id' nor 'id' — cannot join.")
            return None

    for c in ["text", "title", "description"]:
        if c not in ma.columns:
            ma[c] = ""

    ma["__joined_text"] = (
        ma[["text", "title", "description"]]
        .fillna("")
        .astype(str)
        .apply(lambda row: " — ".join([s for s in row if s.strip() and s.strip().lower() != "nan"]).strip(" —"), axis=1)
    )

    keep_cols = ["article_id", "__joined_text", "language", "dominant_industry"]
    keep_cols = [c for c in keep_cols if c in ma.columns]
    ma_small = ma[keep_cols].copy()

    df = gold.merge(ma_small, on="article_id", how="left")

    if "text" in df.columns and df["text"].notna().any():
        df["text"] = df["text"].astype(str)
    else:
        df["text"] = df["__joined_text"].astype(str)

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    too_short = df["text"].str.len() < 5
    if too_short.any():
        n_drop = int(too_short.sum())
        if n_drop > 0:
            print(f"[WARN] Dropping {n_drop} gold rows with missing/too-short text after join.")
        df = df[~too_short].copy()

    if df.empty:
        print("[ERROR] No eligible rows after joining gold with MarketAux text.")
        return None

    if df["y"].dtype == object:
        map_y = {"negative": 0, "neg": 0, "0": 0,
                 "neutral": 1, "neu": 1, "1": 1,
                 "positive": 2, "pos": 2, "2": 2}
        df["y"] = df["y"].astype(str).str.lower().map(map_y)
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)

    if "dominant_industry" in df.columns:
        df["industry"] = df["dominant_industry"]
    elif "industry" in df.columns:
        df["industry"] = df["industry"]
    else:
        df["industry"] = "Unknown"

    if "language" in df.columns:
        df["language"] = df["language"].fillna("unk").astype(str).str.lower()
    else:
        df["language"] = "unk"

    for c in ["rationale", "scope"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df[["article_id", "text", "y", "industry", "language"] + 
              [c for c in ["rationale", "scope"] if c in df.columns]]

def _try_vader(texts, pos_thr=0.05, neg_thr=-0.05):
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        import nltk
        try: nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError: nltk.download("vader_lexicon")
        sia = SentimentIntensityAnalyzer()
        labs = []
        for t in texts:
            c = sia.polarity_scores(t or "")["compound"]
            labs.append(2 if c >= pos_thr else 0 if c <= neg_thr else 1)
        return labs, "VADER"
    except Exception as e:
        print(f"[WARN] VADER unavailable: {e}")
        return None, None

def _lm_label(texts):
    pos_p = Path("data/lexicons/lm/lm_positive.txt")
    neg_p = Path("data/lexicons/lm/lm_negative.txt")
    if not (pos_p.exists() and neg_p.exists()): 
        return None, None
    pos = set(w.strip().lower() for w in pos_p.read_text(encoding="utf-8").splitlines() if w.strip())
    neg = set(w.strip().lower() for w in neg_p.read_text(encoding="utf-8").splitlines() if w.strip())
    tok_re = re.compile(r"[A-Za-z]+")
    labs = []
    for t in texts:
        toks = [x.lower() for x in tok_re.findall(t or "")]
        p = sum((z in pos) for z in toks)
        n = sum((z in neg) for z in toks)
        sc = p - n
        labs.append(2 if sc >= 1 else 0 if sc <= -1 else 1)
    return labs, "LM"

# ----------------------------
# Modeling helpers
# ----------------------------
def _fit_shared_vocab_vectorizer(texts_train):
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=1)
    vec.fit(texts_train)
    return vec

def _fit_lr_on_vec(vec, texts, y):
    X = vec.transform(texts)
    clf = LogisticRegression(max_iter=200, C=2.0, solver="lbfgs", random_state=SEED)
    clf.fit(X, y)
    return clf

def _predict_on_vec(vec, clf, texts):
    X = vec.transform(texts)
    yhat = clf.predict(X)
    proba = clf.predict_proba(X)
    return yhat, proba, X

def _metrics(y_true, y_pred):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class": {str(i): float(s) for i, s in enumerate(
            f1_score(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0)
        )},
        "n": int(len(y_true)),
    }

# ----------------------------
# SHAP (new API with fallback)
# ----------------------------
def _compute_shap_explanation(clf, X_eval, X_bg):
    """
    Returns a tuple (mode, values)
      - If mode == "new": values shape (N, F, C) or (N, F)
      - If mode == "old": values is a list length C with arrays (N, F)
    """
    import shap
    # downsample background for speed/stability
    nbg = min(512, X_bg.shape[0])
    if X_bg.shape[0] > nbg:
        idx = np.linspace(0, X_bg.shape[0]-1, num=nbg, dtype=int)
        X_bg_small = X_bg[idx]
    else:
        X_bg_small = X_bg

    # Try new API first
    try:
        masker = shap.maskers.Independent(X_bg_small)
        explainer = shap.explainers.Linear(clf, masker=masker)
        exp = explainer(X_eval)
        return "new", exp.values
    except Exception:
        pass

    # Fallback to old API with interventional perturbation (valid options)
    try:
        explainer = shap.LinearExplainer(clf, X_bg_small, feature_perturbation="interventional")
        vals = explainer.shap_values(X_eval)
        return "old", vals
    except Exception as e:
        raise RuntimeError(f"SHAP failed: {e}")

def _shap_top(vec, clf, X_eval, X_bg, topk=100):
    try:
        mode, vals = _compute_shap_explanation(clf, X_eval, X_bg)
    except Exception as e:
        print(f"[WARN] SHAP failed: {e}")
        return None

    feats = np.array(vec.get_feature_names_out())
    rows = []

    if mode == "old":
        # list-of-classes -> (N,F) per class
        for c, s in enumerate(vals):
            mean_abs = np.abs(s).mean(axis=0)
            order = np.argsort(-mean_abs)[:topk]
            for r, j in enumerate(order, 1):
                rows.append({"class": c, "rank": r, "token": feats[j], "mean_abs_shap": float(mean_abs[j])})
    else:
        # "new": values shape (N,F,C) or (N,F)
        if vals.ndim == 2:
            mean_abs = np.abs(vals).mean(axis=0)
            order = np.argsort(-mean_abs)[:topk]
            for r, j in enumerate(order, 1):
                rows.append({"class": 0, "rank": r, "token": feats[j], "mean_abs_shap": float(mean_abs[j])})
        else:
            C = vals.shape[2]
            for c in range(C):
                mean_abs = np.abs(vals[:,:,c]).mean(axis=0)
                order = np.argsort(-mean_abs)[:topk]
                for r, j in enumerate(order, 1):
                    rows.append({"class": c, "rank": r, "token": feats[j], "mean_abs_shap": float(mean_abs[j])})

    return pd.DataFrame(rows)

def _compare_ranks_df(top_gold, top_weak):
    from scipy.stats import spearmanr
    from scipy.spatial.distance import jensenshannon
    rows = []
    for c in sorted(set(top_gold["class"]).union(set(top_weak["class"]))):
        Ag = top_gold[top_gold["class"]==c]
        Aw = top_weak[top_weak["class"]==c]
        union = sorted(set(Ag["token"]).union(set(Aw["token"])))
        if not union:
            rows.append({"class": int(c), "spearman": None, "jaccard@50": 0.0, "jaccard@100": 0.0, "jsd": None})
            continue
        max_rank = max(Ag["rank"].max(), Aw["rank"].max(), 100) + 1
        a = Ag.set_index("token")["rank"].reindex(union).fillna(max_rank).values
        b = Aw.set_index("token")["rank"].reindex(union).fillna(max_rank).values
        try: rho, _ = spearmanr(a, b)
        except Exception: rho = None
        a50 = set(Ag.head(50)["token"].tolist()); b50 = set(Aw.head(50)["token"].tolist())
        a100= set(Ag.head(100)["token"].tolist()); b100= set(Aw.head(100)["token"].tolist())
        j50 = float(len(a50 & b50) / max(1, len(a50 | b50)))
        j100= float(len(a100 & b100) / max(1, len(a100 | b100)))
        ma = Ag.set_index("token")["mean_abs_shap"].reindex(union).fillna(0.0).values
        mb = Aw.set_index("token")["mean_abs_shap"].reindex(union).fillna(0.0).values
        pa = ma / (ma.sum() if ma.sum() > 0 else 1.0)
        pb = mb / (mb.sum() if mb.sum() > 0 else 1.0)
        try: jsd = float(jensenshannon(pa, pb, base=2.0)**2)
        except Exception: jsd = None
        rows.append({"class": int(c), "spearman": None if rho is None else float(rho), "jaccard@50": j50, "jaccard@100": j100, "jsd": jsd})
    df = pd.DataFrame(rows)
    macro = {
        "macro_spearman": float(df["spearman"].dropna().mean()) if df["spearman"].notna().any() else None,
        "macro_jaccard@50": float(df["jaccard@50"].mean()) if not df.empty else 0.0,
        "macro_jaccard@100": float(df["jaccard@100"].mean()) if not df.empty else 0.0,
        "macro_jsd": float(df["jsd"].dropna().mean()) if df["jsd"].notna().any() else None,
    }
    return df, macro

# ----------------------------
# Faithfulness (deletion)
# ----------------------------
def _token_delete(text, tokens_to_remove, k):
    if not tokens_to_remove: return text
    toks = [re.escape(t) for t in tokens_to_remove[:k] if t and t.strip()]
    if not toks: return text
    pat = r"\b(" + "|".join(toks) + r")\b"
    return re.sub(pat, " ", text, flags=re.IGNORECASE).replace("  ", " ")

def _faithfulness_curves(vec, clf, texts, y_pred, X_bg, ks=(1,3,5,10)):
    try:
        import shap
    except Exception as e:
        print(f"[WARN] SHAP not available for faithfulness: {e}")
        return None

    X = vec.transform(texts)

    # get SHAP values once
    try:
        mode, vals = _compute_shap_explanation(clf, X, X_bg)
    except Exception as e:
        print(f"[WARN] SHAP for faithfulness failed: {e}")
        return None

    feats = np.array(vec.get_feature_names_out())
    proba = clf.predict_proba(X)
    ks = list(ks)
    rows = []

    def _contrib_for(i, c):
        if mode == "old":
            return np.abs(vals[c][i])
        else:
            if vals.ndim == 2:
                return np.abs(vals[i])
            else:
                return np.abs(vals[i,:,c])

    for i in range(X.shape[0]):
        c = int(y_pred[i])
        contrib = _contrib_for(i, c)
        order = np.argsort(-contrib)
        top_tokens = feats[order].tolist()

        base_prob = float(proba[i, c])
        base_text = texts[i]

        for k in ks:
            new_text = _token_delete(base_text, top_tokens, k)
            X_new = vec.transform([new_text])
            new_proba = clf.predict_proba(X_new)[0]
            new_yhat = int(np.argmax(new_proba))
            prob_drop = float(base_prob - new_proba[c])
            flipped = int(new_yhat != c)
            rows.append({"i": i, "k": k, "flipped": flipped, "prob_drop": prob_drop})

    df = pd.DataFrame(rows)
    curve = df.groupby("k").agg(flip_rate=("flipped","mean"), mean_prob_drop=("prob_drop","mean")).reset_index()
    return curve

# ----------------------------
# Main
# ----------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IND_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_gold()
    if df is None or df.empty:
        return

    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["y"])

    yw, src = _try_vader(tr["text"].tolist())
    if yw is None:
        yw, src = _lm_label(tr["text"].tolist())
    if yw is None:
        print("[ERROR] Neither VADER nor LM weak labels are available.")
        return
    tr = tr.copy(); tr["y_weak"] = yw

    vec = _fit_shared_vocab_vectorizer(tr["text"].tolist())

    gold_clf = _fit_lr_on_vec(vec, tr["text"].tolist(), tr["y"].astype(int).values)
    weak_clf = _fit_lr_on_vec(vec, tr["text"].tolist(), tr["y_weak"].astype(int).values)

    yhat_gold, proba_gold, X_te = _predict_on_vec(vec, gold_clf, te["text"].tolist())
    yhat_weak, proba_weak, _    = _predict_on_vec(vec, weak_clf, te["text"].tolist())

    m_gold = _metrics(te["y"].astype(int).values, yhat_gold)
    m_weak = _metrics(te["y"].astype(int).values, yhat_weak)
    overall = {
        "label_source_weak": src,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "gold_metrics": m_gold,
        "weak_metrics": m_weak,
        "deltas": {
            "acc": float(m_gold["acc"] - m_weak["acc"]),
            "macro_f1": float(m_gold["macro_f1"] - m_weak["macro_f1"]),
        }
    }

    # ---- SHAP tops & divergence on TEST
    # Build background from TRAIN (sampled)
    X_bg_train = vec.transform(tr["text"].tolist())
    try:
        top_gold = _shap_top(vec, gold_clf, X_te, X_bg_train, topk=100)
        top_weak = _shap_top(vec, weak_clf, X_te, X_bg_train, topk=100)
    except Exception as e:
        top_gold, top_weak = None, None
        print(f"[WARN] SHAP top tokens failed: {e}")

    if top_gold is not None and top_weak is not None:
        corr_df, macro = _compare_ranks_df(top_gold, top_weak)
        corr_df.to_csv(OUT_DIR / "shap_rank_corr.csv", index=False)
        overall["shap_divergence_macro"] = macro
    else:
        overall["shap_divergence_macro"] = None

    # ---- Faithfulness (deletion) on TEST
    try:
        curve_gold = _faithfulness_curves(vec, gold_clf, te["text"].tolist(), yhat_gold, X_bg=X_bg_train)
        curve_weak = _faithfulness_curves(vec, weak_clf, te["text"].tolist(), yhat_weak, X_bg=X_bg_train)
        if curve_gold is not None:
            curve_gold.to_csv(OUT_DIR / "flip_curves_gold.csv", index=False)
        if curve_weak is not None:
            curve_weak.to_csv(OUT_DIR / "flip_curves_weak.csv", index=False)
        if curve_gold is not None and curve_weak is not None:
            fig = plt.figure(figsize=(5.2,3.8))
            ks = curve_gold["k"].tolist()
            plt.plot(ks, curve_gold["flip_rate"].tolist(), marker="o", label="GOLD flip-rate")
            plt.plot(ks, curve_weak["flip_rate"].tolist(), marker="o", label="WEAK flip-rate")
            plt.xlabel("Top-k tokens deleted")
            plt.ylabel("Flip rate")
            plt.title("Faithfulness (deletion) — GOLD vs WEAK")
            plt.legend()
            plt.tight_layout()
            plt.savefig(OUT_DIR / "token_flip_curves.png", dpi=160)
            plt.close(fig)
    except Exception as e:
        print(f"[WARN] Faithfulness curves failed: {e}")

    # ---- Per-industry metrics (if present)
    per_ind = []
    if "industry" in te.columns:
        for ind, d in te.groupby("industry"):
            mask = te["industry"] == ind
            yg = te.loc[mask, "y"].astype(int).values
            hg = np.array(yhat_gold)[mask.values]
            hw = np.array(yhat_weak)[mask.values]
            if len(yg) >= 20:
                mg = _metrics(yg, hg); mw = _metrics(yg, hw)
                per = {
                    "industry": ind,
                    "n": int(len(yg)),
                    "gold_acc": mg["acc"], "gold_macro_f1": mg["macro_f1"],
                    "weak_acc": mw["acc"], "weak_macro_f1": mw["macro_f1"],
                    "delta_acc": float(mg["acc"] - mw["acc"]),
                    "delta_macro_f1": float(mg["macro_f1"] - mw["macro_f1"])
                }
                per_ind.append(per)
                (IND_DIR / f"{_slug(ind)}_metrics.json").write_text(json.dumps(per, indent=2))

    (OUT_DIR / "metrics_overall.json").write_text(json.dumps(overall, indent=2))
    if per_ind:
        pd.DataFrame(per_ind).to_csv(OUT_DIR / "metrics_by_industry.csv", index=False)

    print(f"[DONE] Bias audit complete → {OUT_DIR}")

if __name__ == "__main__":
    main()
