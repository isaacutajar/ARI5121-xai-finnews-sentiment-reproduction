# scripts/10_shap_regime_shift_all.py
"""
SHAP regime-shift analysis (time & volatility) across all industries.

Train a single LR+TFIDF per industry on weak labels (VADER → LM fallback),
then compare SHAP token importance between:
  (A) Early vs Late (split by median published date)
  (B) Low-vol vs High-vol days (by sector ETF rolling vol; if ETF CSV present)

Reads:
  data/processed/marketaux/marketaux_news_articles.csv
  data/market/etf/{XLC,XLY,XLE,XLF,XLV,XLI,XLRE,XLU,XLK}.csv  (optional per-industry)

Writes (per industry slug → outputs/regime/{slug}/):
  time_top_tokens_early.csv, time_top_tokens_late.csv
  time_compare.json  (spearman, jaccard@50/@100, jsd)
  vol_top_tokens_low.csv, vol_top_tokens_high.csv  (if ETF present)
  vol_compare.json
Global:
  outputs/regime/summary.csv   (#docs, had_volatility, metrics snippets)

Run: python scripts/10_shap_regime_shift_all.py
"""
from pathlib import Path
import json, re
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from _config import (
    MARKETAUX_ARTICLES_CSV, OUTPUTS_DIR, SEED, INDUSTRIES
)

# -----------------------
# Tunables
# -----------------------
MIN_DOCS_PER_INDUSTRY = 1     # include all industries; warn if tiny
MIN_PER_REGIME = 30           # minimum per split (early/late or low/high) to compute SHAP/compare
MAX_SHAP_SAMPLES = 2000       # cap SHAP eval samples per split
BG_SAMPLES = 512              # background sample size for SHAP

OUT_DIR = OUTPUTS_DIR / "regime"
ETF_DIR = Path("data/market/etf")
RNG = np.random.default_rng(SEED)

SECTOR_TO_ETF = {
    "Communication Services": "XLC",
    "Consumer Cyclical":      "XLY",
    "Energy":                 "XLE",
    "Financial Services":     "XLF",
    "Healthcare":             "XLV",
    "Industrials":            "XLI",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
    "Technology":             "XLK",
    # "Services" has no clean SPDR sector ETF; vol analysis will be skipped
}

def _slug(s): 
    return re.sub(r"[^a-z0-9]+","-", (s or "unknown").lower()).strip("-")

def _try_vader(texts, pos_thr=0.05, neg_thr=-0.05):
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        import nltk
        try: nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError: nltk.download("vader_lexicon")
        sia = SentimentIntensityAnalyzer()
        comps = [sia.polarity_scores(t or "")["compound"] for t in texts]
        def lab(c): 
            if c >= pos_thr: return 2
            if c <= neg_thr: return 0
            return 1
        return [lab(c) for c in comps], "VADER"
    except Exception as e:
        print(f"[WARN] VADER unavailable: {e}")
        return None, None

def _lm_label(texts, lex_dir=Path("data/lexicons/lm")):
    pos_p = lex_dir / "lm_positive.txt"
    neg_p = lex_dir / "lm_negative.txt"
    if not (pos_p.exists() and neg_p.exists()):
        return None, None
    pos = set(w.strip().lower() for w in pos_p.read_text(encoding="utf-8").splitlines() if w.strip())
    neg = set(w.strip().lower() for w in neg_p.read_text(encoding="utf-8").splitlines() if w.strip())
    tok_re = re.compile(r"[A-Za-z]+")
    y = []
    for t in texts:
        toks = [x.lower() for x in tok_re.findall(t or "")]
        p = sum((z in pos) for z in toks)
        n = sum((z in neg) for z in toks)
        sc = p - n
        y.append(2 if sc >= 1 else 0 if sc <= -1 else 1)
    return y, "LM"

def _train_lr(df_text_y):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=2)),
        ("lr", LogisticRegression(max_iter=200, C=2.0, n_jobs=1)),
    ])
    pipe.fit(df_text_y["text"].tolist(), df_text_y["y"].astype(int).values)
    return pipe

def _compute_shap(pipe, texts, bg_mat=None, max_samples=MAX_SHAP_SAMPLES):
    try:
        import shap
        vec = pipe.named_steps["tfidf"]; lr = pipe.named_steps["lr"]
        # evaluate on a capped slice for speed
        texts_eval = texts[:max_samples]
        X = vec.transform(texts_eval)
        # background: if none, use an evenly spaced slice from X
        if bg_mat is None:
            if X.shape[0] == 0:
                return None, None
            idx = np.linspace(0, X.shape[0]-1, num=min(BG_SAMPLES, X.shape[0]), dtype=int)
            bg_mat = X[idx]
        # Try newer & older SHAP APIs gracefully
        try:
            explainer = shap.LinearExplainer(lr, bg_mat)  # modern versions
        except TypeError:
            try:
                explainer = shap.LinearExplainer(lr, bg_mat, feature_perturbation="interventional")
            except Exception:
                explainer = shap.LinearExplainer(lr, bg_mat, feature_dependence="independent")
        shap_vals = explainer.shap_values(X)
        feat_names = np.array(vec.get_feature_names_out())
        return shap_vals, feat_names
    except Exception as e:
        print(f"[WARN] SHAP failed: {e}")
        return None, None

def _top_df(shap_vals, feat_names, topk=100):
    rows = []
    # shap_vals: list-of-classes [C] x (N x F)
    for c, sv in enumerate(shap_vals):
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(-mean_abs)[:topk]
        for r, j in enumerate(order, 1):
            rows.append({"class": c, "rank": r, "token": feat_names[j], "mean_abs_shap": float(mean_abs[j])})
    return pd.DataFrame(rows)

def _compare_rank(topA: pd.DataFrame, topB: pd.DataFrame, kA=100, kB=100):
    # union tokens (all classes together)
    A = topA.sort_values(["class","rank"]).groupby("class").head(kA)
    B = topB.sort_values(["class","rank"]).groupby("class").head(kB)
    rows = []
    for c in sorted(set(A["class"]).union(set(B["class"]))):
        a = A[A["class"]==c][["token","rank"]].set_index("token")["rank"]
        b = B[B["class"]==c][["token","rank"]].set_index("token")["rank"]
        union = sorted(set(a.index).union(set(b.index)))
        if not union:
            rows.append({"class": int(c), "spearman": None, "jaccard@50": 0.0, "jaccard@100": 0.0, "jsd": None})
            continue
        # Spearman on aligned ranks (fill missing with max_rank+1)
        max_rank = max((a.max() if len(a) else 0), (b.max() if len(b) else 0), kA, kB) + 1
        va = np.array([a.get(t, max_rank) for t in union])
        vb = np.array([b.get(t, max_rank) for t in union])
        try:
            rho, _ = spearmanr(va, vb)
            rho = float(rho)
        except Exception:
            rho = None
        # Jaccard of top sets
        a50 = set(A[A["class"]==c].head(50)["token"].tolist())
        b50 = set(B[B["class"]==c].head(50)["token"].tolist())
        a100= set(A[A["class"]==c].head(100)["token"].tolist())
        b100= set(B[B["class"]==c].head(100)["token"].tolist())
        j50 = float(len(a50 & b50) / max(1, len(a50 | b50)))
        j100= float(len(a100 & b100) / max(1, len(a100 | b100)))
        # Jensen–Shannon divergence on normalized mean_abs_shap over union
        ma = A[A["class"]==c].set_index("token")["mean_abs_shap"].reindex(union).fillna(0.0).values
        mb = B[B["class"]==c].set_index("token")["mean_abs_shap"].reindex(union).fillna(0.0).values
        pa = ma / (ma.sum() if ma.sum() > 0 else 1.0)
        pb = mb / (mb.sum() if mb.sum() > 0 else 1.0)
        try:
            jsd = float(jensenshannon(pa, pb, base=2.0)**2)  # JS distance^2 = JSD
        except Exception:
            jsd = None
        rows.append({"class": int(c), "spearman": rho, "jaccard@50": j50, "jaccard@100": j100, "jsd": jsd})
    # macro means
    df = pd.DataFrame(rows)
    macro = {
        "macro_spearman": float(df["spearman"].dropna().mean()) if df["spearman"].notna().any() else None,
        "macro_jaccard@50": float(df["jaccard@50"].mean()) if not df.empty else 0.0,
        "macro_jaccard@100": float(df["jaccard@100"].mean()) if not df.empty else 0.0,
        "macro_jsd": float(df["jsd"].dropna().mean()) if df["jsd"].notna().any() else None,
        "per_class": rows
    }
    return macro

def _load_etf(symbol):
    p = ETF_DIR / f"{symbol}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["ret"] = df["Adj Close"].pct_change()
    df["rv_21"] = df["ret"].rolling(21).std() * np.sqrt(252)
    df["date"] = df["Date"].dt.date
    return df[["date","rv_21"]]

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not Path(MARKETAUX_ARTICLES_CSV).exists():
        print(f"[ERROR] Missing: {MARKETAUX_ARTICLES_CSV}")
        return

    df = pd.read_csv(MARKETAUX_ARTICLES_CSV)
    if df.empty:
        print("[ERROR] No articles.")
        return
    # Preprocess
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["date"] = df["published_at"].dt.date
    df["language"] = df["language"].fillna("unk").str.lower()
    df["dominant_industry"] = df["dominant_industry"].fillna("").replace({"": "Unknown"})
    df["text"] = df["text"].fillna("")
    df = df[(df["language"]=="en") & (df["text"].str.len()>=15)].copy()

    summary_rows = []

    # Iterate industries present
    inds = sorted(df["dominant_industry"].unique())
    for ind in inds:
        d = df[df["dominant_industry"]==ind].copy()
        n_docs = len(d)
        if n_docs < MIN_DOCS_PER_INDUSTRY:
            print(f"[INFO] Skip industry '{ind}' (n={n_docs} < {MIN_DOCS_PER_INDUSTRY}).")
            continue
        if n_docs < 100:
            print(f"[WARN] Small industry '{ind}' (n={n_docs}); results may be noisy.")

        slug = _slug(ind)
        idir = OUT_DIR / slug
        idir.mkdir(parents=True, exist_ok=True)

        # Weak labels
        y, src = _try_vader(d["text"].tolist())
        if y is None:
            y, src = _lm_label(d["text"].tolist())
        if y is None:
            print(f"[WARN] No weak labels available for '{ind}'. Skipping.")
            continue
        d["y"] = y

        # Need at least 2 classes to train a classifier
        if pd.Series(d["y"]).nunique() < 2:
            print(f"[INFO] '{ind}' has <2 weak-label classes; skipping.")
            summary_rows.append({
                "industry": ind, "slug": slug, "n_docs_en": int(n_docs),
                "weak_label_source": src, "has_volatility_etf": False,
                "time_macro_spearman": None, "vol_macro_spearman": None
            })
            continue

        # Train LR
        pipe = _train_lr(d[["text","y"]])

        # Prepare SHAP background on a subset
        try:
            vec = pipe.named_steps["tfidf"]
            X_all = vec.transform(d["text"].tolist())
            bg_idx = np.linspace(0, X_all.shape[0]-1, num=min(BG_SAMPLES, X_all.shape[0]), dtype=int)
            bg = X_all[bg_idx]
        except Exception:
            bg = None

        # ---- Time regimes: early vs late by median date
        time_macro_spr = None
        if d["published_at"].notna().any():
            median_ts = d["published_at"].median()
            early = d[d["published_at"] <= median_ts]
            late  = d[d["published_at"] >  median_ts]
            if len(early) >= MIN_PER_REGIME and len(late) >= MIN_PER_REGIME:
                shap_A, feats = _compute_shap(pipe, early["text"].tolist(), bg_mat=bg)
                shap_B, _     = _compute_shap(pipe, late["text"].tolist(),  bg_mat=bg)
                if shap_A is not None and shap_B is not None:
                    topA = _top_df(shap_A, feats, topk=100)
                    topB = _top_df(shap_B, feats, topk=100)
                    topA.to_csv(idir / "time_top_tokens_early.csv", index=False, encoding="utf-8-sig")
                    topB.to_csv(idir / "time_top_tokens_late.csv",  index=False, encoding="utf-8-sig")
                    comp = _compare_rank(topA, topB, 100, 100)
                    (idir / "time_compare.json").write_text(json.dumps(comp, indent=2))
                    time_macro_spr = comp.get("macro_spearman")
            else:
                print(f"[INFO] '{ind}' time split too small (early={len(early)}, late={len(late)}; need ≥{MIN_PER_REGIME}).")
        else:
            print(f"[INFO] '{ind}' has no valid timestamps; skipping time split.")

        # ---- Volatility regimes: low vs high by ETF rolling vol tertiles
        etf = SECTOR_TO_ETF.get(ind)
        had_vol = False
        vol_macro_spr = None
        if etf:
            edf = _load_etf(etf)
            if edf is not None and not edf["rv_21"].isna().all():
                had_vol = True
                q_low, q_high = edf["rv_21"].quantile([0.33, 0.67]).values
                edf["vol_regime"] = np.where(edf["rv_21"]<=q_low, "low",
                                       np.where(edf["rv_21"]>=q_high, "high", "mid"))
                m = d.merge(edf[["date","vol_regime"]], on="date", how="left")
                low  = m[m["vol_regime"]=="low"]
                high = m[m["vol_regime"]=="high"]
                if len(low) >= MIN_PER_REGIME and len(high) >= MIN_PER_REGIME:
                    shap_L, feats = _compute_shap(pipe, low["text"].tolist(),  bg_mat=bg)
                    shap_H, _     = _compute_shap(pipe, high["text"].tolist(), bg_mat=bg)
                    if shap_L is not None and shap_H is not None:
                        topL = _top_df(shap_L, feats, topk=100)
                        topH = _top_df(shap_H, feats, topk=100)
                        topL.to_csv(idir / "vol_top_tokens_low.csv",  index=False, encoding="utf-8-sig")
                        topH.to_csv(idir / "vol_top_tokens_high.csv", index=False, encoding="utf-8-sig")
                        comp = _compare_rank(topL, topH, 100, 100)
                        (idir / "vol_compare.json").write_text(json.dumps(comp, indent=2))
                        vol_macro_spr = comp.get("macro_spearman")
                else:
                    print(f"[INFO] '{ind}' vol split too small (low={len(low)}, high={len(high)}; need ≥{MIN_PER_REGIME}).")
            else:
                print(f"[INFO] ETF data missing/empty for '{ind}' ({etf}); skipping vol split.")
        else:
            print(f"[INFO] No ETF mapping for '{ind}'; skipping vol split.")

        # Summary row
        summary_rows.append({
            "industry": ind,
            "slug": slug,
            "n_docs_en": int(n_docs),
            "weak_label_source": src,
            "has_volatility_etf": bool(had_vol),
            "time_macro_spearman": time_macro_spr,
            "vol_macro_spearman": vol_macro_spr
        })
        print(f"[OK] {ind}: time rho={time_macro_spr} | vol rho={vol_macro_spr} | n={n_docs}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(OUT_DIR / "summary.csv", index=False)
        print(f"[DONE] Wrote summary → {OUT_DIR/'summary.csv'}")
    else:
        print("[WARN] No industries processed. Check inputs/dependencies.")

if __name__ == "__main__":
    main()
