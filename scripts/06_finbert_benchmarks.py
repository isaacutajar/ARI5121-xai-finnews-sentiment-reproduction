# scripts/06_finbert_benchmarks.py
"""
FinBERT inference on FPB & FiQA.

Model: ProsusAI/finbert (labels: positive/negative/neutral)
Outputs per dataset:
  outputs/baselines/finbert/{ds}_metrics.json
  outputs/baselines/finbert/{ds}_predictions.csv
Run: python scripts/06_finbert_benchmarks.py
"""
from pathlib import Path
import json, math
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from _config import FPB_CSV, FIQA_HEADLINES_CSV, OUTPUTS_DIR

OUT_DIR = OUTPUTS_DIR / "baselines" / "finbert"

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

def _map_label(lbl: str) -> int:
    s = (lbl or "").lower()
    if "pos" in s: return 2
    if "neg" in s: return 0
    return 1

def _infer_finbert(texts):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
    except Exception as e:
        print(f"[ERROR] transformers not installed: {e}")
        return None

    try:
        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, return_all_scores=False, truncation=True)
    except Exception as e:
        print(f"[ERROR] Could not load FinBERT: {e}")
        return None

    preds = []
    B = 32
    for i in range(0, len(texts), B):
        chunk = texts[i:i+B]
        out = pipe(chunk)
        for o in out:
            preds.append(_map_label(o["label"]))
    return preds

def _run(name: str, df: pd.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = _infer_finbert(df["text"].tolist())
    if y_pred is None:
        print(f"[WARN] FinBERT inference skipped for {name}.")
        return
    metr = _metrics(df["y"].tolist(), y_pred)
    (OUT_DIR / f"{name}_metrics.json").write_text(json.dumps(metr, indent=2))
    pd.DataFrame({"text": df["text"], "y": df["y"], "y_pred": y_pred}).to_csv(
        OUT_DIR / f"{name}_predictions.csv", index=False, encoding="utf-8-sig"
    )
    print(f"[OK] FinBERT done for {name}")

def main():
    if FPB_CSV.exists(): _run("fpb", _load(FPB_CSV))
    else: print(f"[WARN] Missing {FPB_CSV}")
    if FIQA_HEADLINES_CSV.exists(): _run("fiqa", _load(FIQA_HEADLINES_CSV))
    else: print(f"[WARN] Missing {FIQA_HEADLINES_CSV}")
    print("[DONE] FinBERT benchmarks.")

if __name__ == "__main__":
    main()
