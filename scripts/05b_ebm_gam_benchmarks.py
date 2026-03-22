# scripts/05b_ebm_gam_benchmarks.py
"""
Glass-box baselines: EBM (Explainable Boosting Machine). (GAM optional.)

Vectorization: TF-IDF (<=3000 features) → dense → EBM (multiclass).
Outputs per dataset:
  outputs/baselines/ebm_gam/{ds}_metrics.json
  outputs/baselines/ebm_gam/{ds}_feat_importance.csv
  outputs/baselines/ebm_gam/{ds}_predictions.csv
Run: python scripts/05b_ebm_gam_benchmarks.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from _config import FPB_CSV, FIQA_HEADLINES_CSV, OUTPUTS_DIR, SEED

OUT_DIR = OUTPUTS_DIR / "baselines" / "ebm_gam"

def _load(path: Path):
    df = pd.read_csv(path).dropna(subset=["text","y"]).copy()
    df["y"] = df["y"].astype(int)
    return df

def _metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "f1_per_class": {str(k): float(v) for k, v in zip([0,1,2], f1_score(y_true, y_pred, average=None, labels=[0,1,2]))},
        "confusion_matrix": cm.tolist(),
    }

def _ebm_term_importances(ebm):
    """
    Handle interpret version differences:
      - older: ebm.term_importances_  (ndarray-like)
      - newer: ebm.term_importances   (property) or ebm.term_importances() (callable)
    Returns a 1D numpy array.
    """
    imp = None
    if hasattr(ebm, "term_importances_"):
        imp = getattr(ebm, "term_importances_")
    elif hasattr(ebm, "term_importances"):
        attr = getattr(ebm, "term_importances")
        imp = attr() if callable(attr) else attr
    if imp is None:
        # very defensive fallback via global explanation (slower)
        try:
            exp = ebm.explain_global()
            data = exp.data()
            # interpret usually stores overall scores under 'scores'
            imp = np.array(data.get("scores") or data.get("overall", {}).get("scores"))
        except Exception:
            imp = None
    if imp is None:
        raise RuntimeError("Could not obtain EBM term importances from this interpret version.")
    return np.asarray(imp).ravel()

def _run_ebm(name: str, df: pd.DataFrame):
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        from interpret import show  # noqa: F401 (kept for completeness)
    except Exception as e:
        print(f"[WARN] interpret not installed ({e}). Skipping EBM for {name}. pip install interpret")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["y"].values, test_size=0.2, random_state=SEED, stratify=df["y"]
    )

    vec = TfidfVectorizer(ngram_range=(1,2), max_features=3000, min_df=2)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    # Dense (EBM expects dense ndarray)
    Xtr = Xtr.toarray()
    Xte = Xte.toarray()

    ebm = ExplainableBoostingClassifier(
        interactions=0,        # pure GAM-like (no pairwise interactions)
        max_bins=64,
        learning_rate=0.05,
        max_leaves=3,
        validation_size=0.15,
        random_state=SEED
    )
    ebm.fit(Xtr, y_train)
    y_pred = ebm.predict(Xte)
    metr = _metrics(y_test, y_pred)
    (OUT_DIR / f"{name}_metrics.json").write_text(json.dumps(metr, indent=2))

    # feature importances aligned to vectorizer features
    feats = vec.get_feature_names_out()
    imp = _ebm_term_importances(ebm)

    # interpret's "terms" align to input features when interactions=0, but be defensive:
    if len(imp) != len(feats):
        # pad/truncate to match; this should rarely happen
        m = min(len(imp), len(feats))
        imp = imp[:m]
        feats = feats[:m]

    df_imp = pd.DataFrame({"feature": feats, "importance": imp}).sort_values("importance", ascending=False)
    df_imp.to_csv(OUT_DIR / f"{name}_feat_importance.csv", index=False, encoding="utf-8-sig")

    # predictions (keep original text for readability)
    pd.DataFrame({"text": X_test, "y": y_test, "y_pred": y_pred}).to_csv(
        OUT_DIR / f"{name}_predictions.csv", index=False, encoding="utf-8-sig"
    )
    print(f"[OK] EBM finished for {name}")

def main():
    tasks = []
    if FPB_CSV.exists(): tasks.append(("fpb", _load(FPB_CSV)))
    else: print(f"[WARN] Missing {FPB_CSV}")
    if FIQA_HEADLINES_CSV.exists(): tasks.append(("fiqa", _load(FIQA_HEADLINES_CSV)))
    else: print(f"[WARN] Missing {FIQA_HEADLINES_CSV}")

    for name, df in tasks:
        _run_ebm(name, df)

    print("[DONE] EBM/GAM baselines (GAM optional; add later if needed).")

if __name__ == "__main__":
    main()
