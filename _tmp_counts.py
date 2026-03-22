import pandas as pd, re
from pathlib import Path

PREDS = pd.read_csv(Path("outputs/finetune_eval/predictions.csv"))
for c in list(PREDS.columns):
    if c in ["label","true_label","gold","y_true"]:
        PREDS.rename(columns={c:"y"}, inplace=True)
    if c in ["pred","prediction","predicted","pred_label","predicted_label","y_pred"]:
        PREDS.rename(columns={c:"y_pred"}, inplace=True)

errs = PREDS[PREDS["y"] != PREDS["y_pred"]].copy()
NW = {"not","no","never","without","neither","nor"}

def cat(r):
    tokens = set(re.findall(r"\w+", str(r["text"]).lower()))
    if tokens & NW:
        return "Negation scope"
    elif abs(int(r["y"]) - int(r["y_pred"])) == 2:
        return "Polarity reversal (Neg<->Pos)"
    elif int(r["y"]) == 1 or int(r["y_pred"]) == 1:
        return "Neutral boundary confusion"
    else:
        return "Other"

errs["cat"] = errs.apply(cat, axis=1)
vc = errs["cat"].value_counts()
n = len(errs)
for k, v in vc.items():
    print(f"{k}: {v} ({v/n*100:.1f}%)")
print(f"Total errors: {n} / {len(PREDS)} ({n/len(PREDS)*100:.1f}%)")
