# scripts/07_marketaux_weaklabel_lr.py
"""
Weak-label LR (TF-IDF) on Marketaux EN news across all industries.
Labels via VADER (compound ≥ pos_thr → pos / ≤ neg_thr → neg / else neu).
Falls back to LM lists if VADER unavailable.

Outputs:
 models/weaklr/lr_tfidf_marketaux_overall.joblib
 models/weaklr/lr_tfidf_marketaux_{slug}.joblib     (if trained)
 outputs/weaklabel_lr/overall_{metrics,predictions,shap_top_tokens}.*
 outputs/weaklabel_lr/by_industry/{slug}_{metrics,predictions,shap_top_tokens}.*
 outputs/weaklabel_lr/summary.json

Run: python scripts/07_marketaux_weaklabel_lr.py
"""
from pathlib import Path
import re, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib

from _config import (
  MARKETAUX_ARTICLES_CSV, OUTPUTS_DIR, MODELS_DIR, SEED,
  VADER_POS_THR, VADER_NEG_THR, LEXICON_DIR
)

OUT_DIR = OUTPUTS_DIR / "weaklabel_lr"
OUT_INDS = OUT_DIR / "by_industry"
MODELS_WEAKLR = MODELS_DIR / "weaklr"
TOKEN_RE = re.compile(r"[A-Za-z]+")
# minimal size to *train* a dedicated per-industry model; below this we use the overall model
MIN_TRAIN_N = 20

def _slug(s: str) -> str:
  return re.sub(r"[^a-z0-9]+","-", (s or "unknown").lower()).strip("-")

def _try_vader(texts):
  try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    try: nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError: nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()
  except Exception as e:
    print(f"[WARN] VADER unavailable ({e}). Will try LM fallback.")
    return None, None
  comps = [sia.polarity_scores(t or "")["compound"] for t in texts]
  def lab(c):
    if c >= VADER_POS_THR: return 2
    if c <= VADER_NEG_THR: return 0
    return 1
  labels = [lab(c) for c in comps]
  return comps, labels

def _load_lm_sets():
  pos_p = LEXICON_DIR / "lm_positive.txt"
  neg_p = LEXICON_DIR / "lm_negative.txt"
  if not (pos_p.exists() and neg_p.exists()):
    return None, None
  pos = set(w.strip().lower() for w in pos_p.read_text(encoding="utf-8").splitlines() if w.strip())
  neg = set(w.strip().lower() for w in neg_p.read_text(encoding="utf-8").splitlines() if w.strip())
  return pos, neg

def _lm_label(texts):
  pos, neg = _load_lm_sets()
  if pos is None:
    print("[ERROR] LM lists missing and VADER unavailable. Place LM files or install nltk.")
    return None
  labels = []
  for t in texts:
    toks = [x.lower() for x in TOKEN_RE.findall(t or "")]
    p = sum(1 for z in toks if z in pos)
    n = sum(1 for z in toks if z in neg)
    sc = p - n
    if sc >= 1: labels.append(2)
    elif sc <= -1: labels.append(0)
    else: labels.append(1)
  return labels

def _metrics(y_true, y_pred):
  cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
  return {
    "acc": float(accuracy_score(y_true, y_pred)),
    "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    "f1_per_class": {str(k): float(v) for k, v in zip([0,1,2], f1_score(y_true, y_pred, average=None, labels=[0,1,2]))},
    "confusion_matrix": cm.tolist(),
    "n": int(len(y_true)),
  }

def _shap_values_to_list(shap_obj):
  """
  Normalize SHAP outputs to a list [class0, class1, class2],
  each item: (n_samples, n_features).
  Works across SHAP versions.
  """
  import numpy as np
  # New API (Explanation)
  if hasattr(shap_obj, "values"):
    arr = shap_obj.values
    if isinstance(arr, np.ndarray) and arr.ndim == 3:
      return [arr[:, :, c] for c in range(arr.shape[2])]
    elif isinstance(arr, np.ndarray) and arr.ndim == 2:
      return [arr]
  # Old API (list or ndarray)
  if isinstance(shap_obj, list):
    return shap_obj
  if isinstance(shap_obj, np.ndarray):
    if shap_obj.ndim == 3:
      return [shap_obj[:, :, c] for c in range(shap_obj.shape[2])]
    if shap_obj.ndim == 2:
      return [shap_obj]
  return []

def _top_tokens_from_shap(shap_vals, feature_names, topk=100):
  rows = []
  for c, sv in enumerate(shap_vals):
    if sv.size == 0: # safety
      continue
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)[:topk]
    for rank, j in enumerate(order, 1):
      rows.append({"class": c, "rank": rank, "token": feature_names[j], "mean_abs_shap": float(mean_abs[j])})
  return pd.DataFrame(rows)

def _write_predictions_file(path: Path, meta_df: pd.DataFrame, y_pred, proba, classes_):
  """Write predictions with stable proba_0/1/2 columns, filling missing with NaN."""
  n = len(meta_df)
  proba_full = np.full((n, 3), np.nan, dtype=float)
  # classes_ are the ones present in the fitted LR (e.g., [0,1] for binary)
  class_to_idx = {int(c): i for i, c in enumerate(classes_)}
  for target_c in [0,1,2]:
    if target_c in class_to_idx:
      proba_full[:, target_c] = proba[:, class_to_idx[target_c]]
  out = meta_df.copy()
  out["y_weak"] = meta_df["y_weak"]
  out["y_pred"] = y_pred
  out["proba_0"] = proba_full[:, 0]
  out["proba_1"] = proba_full[:, 1]
  out["proba_2"] = proba_full[:, 2]
  out.to_csv(path, index=False, encoding="utf-8-sig")

def _fit_lr_pipeline(Xtr, ytr):
  pipe = Pipeline([
        # --- THIS IS THE ONLY CHANGE ---
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=2, stop_words='english')),
    ("lr", LogisticRegression(max_iter=200, C=2.0, solver="lbfgs")),
  ])
  pipe.fit(Xtr, ytr)
  return pipe

def _shap_export(name: str, pipe: Pipeline, X_background_texts, X_explain_texts, out_dir: Path):
  """Compute and export top tokens via SHAP with interventional background."""
  try:
    import shap, numpy as np # noqa
    vec = pipe.named_steps["tfidf"]; lr = pipe.named_steps["lr"]
    Xbg = vec.transform(X_background_texts)
    Xte = vec.transform(X_explain_texts)
    
    # Use the updated 'feature_perturbation' parameter for modern SHAP versions
    try:
      explainer = shap.LinearExplainer(lr, Xbg, feature_perturbation="interventional")
    except TypeError:
            # Fallback for older SHAP versions
      explainer = shap.LinearExplainer(lr, Xbg, feature_dependence="independent")
    
    shap_raw = explainer.shap_values(Xte)
    shap_list = _shap_values_to_list(shap_raw)

    if shap_list:
      feats = np.array(vec.get_feature_names_out())
      top = _top_tokens_from_shap(shap_list, feature_names=feats, topk=100)
      top.to_csv(out_dir / f"{name}_shap_top_tokens.csv", index=False, encoding="utf-8-sig")
      print(f"[OK] SHAP exported for {name}")
    else:
      print(f"[WARN] SHAP returned unexpected shape for {name}; skipping export.")
  except Exception as e:
    print(f"[WARN] SHAP failed for {name}: {e}")

def _train_pack(name: str, texts, y, meta_df, out_dir: Path, save_model=True):
  """
  Train a new LR model; write metrics, predictions, SHAP.
  Returns a bundle {pipe, background_texts} for reuse.
  """
  out_dir.mkdir(parents=True, exist_ok=True)
  MODELS_WEAKLR.mkdir(parents=True, exist_ok=True)

  # choose a safe split (no stratify if it would fail)
  unique, counts = np.unique(y, return_counts=True)
  can_stratify = len(unique) >= 2 and counts.min() >= 2
  tsz = 0.2 if len(y) >= 8 else 0.5 # with tiny sets, make a larger test split
  if can_stratify:
    Xtr, Xte, ytr, yte, mtr, mte = train_test_split(
      texts, y, meta_df, test_size=tsz, random_state=SEED, stratify=y
    )
  else:
    Xtr, Xte, ytr, yte, mtr, mte = train_test_split(
      texts, y, meta_df, test_size=tsz, random_state=SEED
    )

  if len(np.unique(ytr)) < 2:
    # cannot fit a classifier with a single class in train
    # write degenerate outputs using majority class
    maj = int(pd.Series(y).mode().iloc[0])
    yhat = np.array([maj] * len(yte))
    metr = _metrics(yte, yhat)
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(metr, indent=2))
    # predictions on full set (degenerate)
    y_full = np.array([maj] * len(meta_df))
    proba = np.zeros((len(meta_df), 1)) # no real probs
    _write_predictions_file(out_dir / f"{name}_predictions.csv", meta_df, y_full, proba, classes_=[maj])
    print(f"[WARN] Degenerate training set for {name} (single class). Wrote baseline outputs.")
    return {"pipe": None, "background_texts": None}

  pipe = _fit_lr_pipeline(Xtr, ytr)

  # eval on holdout
  yhat = pipe.predict(Xte)
  metr = _metrics(yte, yhat)
  (out_dir / f"{name}_metrics.json").write_text(json.dumps(metr, indent=2))

  # predictions on full set
  vec = pipe.named_steps["tfidf"]
  y_full = pipe.predict(meta_df["text"].tolist())
  proba_full = pipe.predict_proba(meta_df["text"].tolist())
  _write_predictions_file(out_dir / f"{name}_predictions.csv", meta_df, y_full, proba_full, classes_=pipe.named_steps["lr"].classes_)

  # save model
  if save_model:
    joblib.dump(pipe, MODELS_WEAKLR / f"lr_tfidf_marketaux_{name}.joblib")

  # SHAP top tokens (use train texts as background, explain on test texts)
  _shap_export(name, pipe, X_background_texts=Xtr, X_explain_texts=Xte, out_dir=out_dir)

  return {"pipe": pipe, "background_texts": Xtr}

def _predict_only_pack(name: str, model_bundle, meta_df, out_dir: Path, background_texts):
  """
  Use an existing (overall) model to generate per-industry outputs.
  """
  out_dir.mkdir(parents=True, exist_ok=True)
  pipe = model_bundle["pipe"]
  if pipe is None:
    # No overall model? fallback: majority label baseline on this slice
    y = meta_df["y_weak"].values
    maj = int(pd.Series(y).mode().iloc[0])
    y_full = np.array([maj] * len(meta_df))
    proba = np.zeros((len(meta_df), 1))
    _write_predictions_file(out_dir / f"{name}_predictions.csv", meta_df, y_full, proba, classes_=[maj])
    metr = _metrics(y, y_full)
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(metr, indent=2))
    return

  # predictions
  y_full = pipe.predict(meta_df["text"].tolist())
  proba_full = pipe.predict_proba(meta_df["text"].tolist())
  _write_predictions_file(out_dir / f"{name}_predictions.csv", meta_df, y_full, proba_full, classes_=pipe.named_steps["lr"].classes_)

  # metrics vs weak labels
  metr = _metrics(meta_df["y_weak"].values, y_full)
  (out_dir / f"{name}_metrics.json").write_text(json.dumps(metr, indent=2))

  # SHAP using overall background; explain on this slice
  _shap_export(name, pipe, X_background_texts=background_texts, X_explain_texts=meta_df["text"].tolist(), out_dir=out_dir)

def main():
  if not MARKETAUX_ARTICLES_CSV.exists():
    print(f"[ERROR] Missing: {MARKETAUX_ARTICLES_CSV}")
    return
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  OUT_INDS.mkdir(parents=True, exist_ok=True)
  MODELS_WEAKLR.mkdir(parents=True, exist_ok=True)

  df = pd.read_csv(MARKETAUX_ARTICLES_CSV)
  df["text"] = df["text"].fillna("")
  df["language"] = df["language"].fillna("unk").str.lower()
  df["dominant_industry"] = df["dominant_industry"].fillna("").replace({"": "Unknown"})
  df = df[df["language"].eq("en") & (df["text"].str.len() >= 15)].copy()
  if df.empty:
    print("[ERROR] No EN rows with text.")
    return

  # weak labels
  comps, y_vader = _try_vader(df["text"].tolist())
  if y_vader is None:
    y_labels = _lm_label(df["text"].tolist())
    label_src = "LM"
  else:
    y_labels = y_vader
    label_src = "VADER"

  df["y_weak"] = y_labels
  df = df.dropna(subset=["y_weak"]).copy()
  df["y_weak"] = df["y_weak"].astype(int)

  # ---- overall model ----
  meta_overall = df[["article_id","published_at","source","url","dominant_industry","language","text","y_weak"]].copy()
  overall_bundle = _train_pack("overall", df["text"].tolist(), df["y_weak"].values, meta_overall, OUT_DIR, save_model=True)

  # ---- per industry (include all; train if feasible, else use overall model) ----
  summary = {"label_source": label_src, "overall_n": int(len(df)), "by_industry": {}}
  for ind, d in df.groupby("dominant_industry"):
    name = f"ind_{_slug(ind)}"
    n = len(d)
    y_unique = d["y_weak"].nunique()
    summary["by_industry"][ind] = {"n": int(n), "unique_labels": int(y_unique)}

    meta = d[["article_id","published_at","source","url","dominant_industry","language","text","y_weak"]].copy()
    out_dir = OUT_INDS

    can_train = (n >= MIN_TRAIN_N) and (y_unique >= 2)
    if can_train:
      _train_pack(name, d["text"].tolist(), d["y_weak"].values, meta, out_dir, save_model=True)
      summary["by_industry"][ind]["trained"] = True
      summary["by_industry"][ind]["used_overall_fallback"] = False
    else:
      # use overall model to still provide outputs for this industry
      _predict_only_pack(name, overall_bundle, meta, out_dir, background_texts=overall_bundle["background_texts"])
      summary["by_industry"][ind]["trained"] = False
      summary["by_industry"][ind]["used_overall_fallback"] = True

  (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
  print("[DONE] Weak-label LR trained. Summary:", summary)

if __name__ == "__main__":
  main()