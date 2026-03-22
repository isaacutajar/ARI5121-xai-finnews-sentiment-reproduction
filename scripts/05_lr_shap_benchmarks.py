# scripts/05_lr_shap_benchmarks.py
"""
LR (TF-IDF) + SHAP on FPB & FiQA.

Writes per-dataset:
  outputs/baselines/lr_shap/{ds}_metrics.json
  outputs/baselines/lr_shap/{ds}_predictions.csv
  outputs/baselines/lr_shap/{ds}_shap_top_tokens.csv
  models/lr_tfidf_{ds}.joblib
Run: python scripts/05_lr_shap_benchmarks.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

from _config import FPB_CSV, FIQA_HEADLINES_CSV, OUTPUTS_DIR, MODELS_DIR, SEED

OUT_DIR = OUTPUTS_DIR / "baselines" / "lr_shap"

def _load(name, path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text","y"]).copy()
    df["y"] = df["y"].astype(int)
    return df

def _metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "f1_per_class": {str(k): float(v) for k, v in zip([0,1,2], 
        f1_score(y_true, y_pred, average=None, labels=[0,1,2]))},
        "confusion_matrix": cm.tolist(),
    }

def _top_tokens_from_shap(shap_vals, feature_names, topk=50):
    # shap_vals is list length n_classes, each [n_samples,n_features]
    rows = []
    for c, sv in enumerate(shap_vals):
        # mean absolute shap per feature
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(-mean_abs)[:topk]
        for rank, j in enumerate(order, 1):
            rows.append({"class": c, "rank": rank, "token": feature_names[j], "mean_abs_shap": float(mean_abs[j])})
    return pd.DataFrame(rows)

def _train_eval(name, df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["y"].values, test_size=0.2, random_state=SEED, stratify=df["y"]
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=2)),
        ("lr", LogisticRegression(max_iter=200, n_jobs=1, C=2.0)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metr = _metrics(y_test, y_pred)
    (OUT_DIR / f"{name}_metrics.json").write_text(json.dumps(metr, indent=2))
    pd.DataFrame({"text": X_test, "y": y_test, "y_pred": y_pred}).to_csv(OUT_DIR / f"{name}_predictions.csv", index=False, encoding="utf-8-sig")

    # save model
    joblib.dump(pipe, MODELS_DIR / f"lr_tfidf_{name}.joblib")

    # SHAP
    try:
        import shap
        vec = pipe.named_steps["tfidf"]
        lr  = pipe.named_steps["lr"]
        X_test_mat = vec.transform(X_test)
        explainer = shap.LinearExplainer(lr, vec.transform(X_train), feature_dependence="independent")
        shap_vals = explainer.shap_values(X_test_mat)
        top = _top_tokens_from_shap(shap_vals, feature_names=np.array(vec.get_feature_names_out()), topk=100)
        top.to_csv(OUT_DIR / f"{name}_shap_top_tokens.csv", index=False, encoding="utf-8-sig")
        print(f"[OK] SHAP exported for {name}")
    except Exception as e:
        print(f"[WARN] SHAP failed for {name}: {e}")

def main():
    tasks = []
    if FPB_CSV.exists(): tasks.append(("fpb", _load("fpb", FPB_CSV)))
    else: print(f"[WARN] Missing {FPB_CSV}")
    if FIQA_HEADLINES_CSV.exists(): tasks.append(("fiqa", _load("fiqa", FIQA_HEADLINES_CSV)))
    else: print(f"[WARN] Missing {FIQA_HEADLINES_CSV}")

    for name, df in tasks:
        print(f"== {name.upper()} ==")
        _train_eval(name, df)
    print("[DONE] LR+SHAP benchmarks.")

if __name__ == "__main__":
    main()
