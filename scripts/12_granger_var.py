# scripts/12_granger_var.py
"""
Granger/VAR tests: does daily sentiment Granger-cause sector ETF returns?

Uses the high-quality sentiment signal from the fine-tuned FinBERT model.

Procedure per industry with ETF:
- Load daily sentiment mean.
- Align with ETF daily returns.
- Standardize both series.
- Fit VAR, lag order by AIC (adaptive cap).
- Test causality: (a) sentiment -> returns, (b) returns -> sentiment

Outputs:
- outputs/granger/<slug>/var_summary.json
- outputs/granger/<slug>/residuals.csv
Global:
- outputs/granger/summary.csv

Run: python scripts/12_granger_var.py
"""
from pathlib import Path
import re, json
import numpy as np
import pandas as pd

# --- Configuration ---
try:
    from _config import OUTPUTS_DIR, SEED
except ImportError:
    ROOT = Path(__file__).resolve().parents[1]
    OUTPUTS_DIR = ROOT / "outputs"
    SEED = 42
    print("Warning: Could not import from _config.py, using fallback paths.")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ETF_DIR = DATA_DIR / "market" / "etf"
OUT_DIR = OUTPUTS_DIR / "granger"
# Path to the new high-quality signal file
SIG_CSV = OUTPUTS_DIR / "finetune_eval" / "daily_sentiment_finetuned.csv"

MIN_OBS = 25
MAX_LAG_CAP = 5
NEAR_CONST_STD = 1e-12

SECTOR_TO_ETF = {
    "Communication Services": "XLC", "Consumer Cyclical": "XLY", "Energy": "XLE",
    "Financial Services": "XLF", "Healthcare": "XLV", "Industrials": "XLI",
    "Real Estate": "XLRE", "Utilities": "XLU", "Technology": "XLK",
}

# --- Helpers ---
def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", (s or "unknown").lower()).strip("-")

def _read_etf(ticker: str):
    p = ETF_DIR / f"{ticker}.csv"
    if not p.exists(): return None
    df = pd.read_csv(p, parse_dates=["Date"])
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["date"] = df["Date"].dt.date
    df["ret"] = pd.to_numeric(df["Adj Close"], errors="coerce").pct_change()
    return df[["date", "ret"]].dropna()

def _load_signal():
    """Simplified function to load only the specified signal file."""
    if not SIG_CSV.exists():
        print(f"[ERROR] Missing sentiment signal file: {SIG_CSV}")
        print("-> Please run '11a_generate_finetuned_sentiment_signal.py' first.")
        return None
    d = pd.read_csv(SIG_CSV, parse_dates=["date"])
    d["date"] = d["date"].dt.date
    print(f"[INFO] Loaded sentiment signal from {SIG_CSV}")
    return d

def _standardize(a: pd.Series):
    a = pd.to_numeric(a, errors="coerce")
    mu, sd = a.mean(), a.std(ddof=0)
    if not np.isfinite(sd) or sd < NEAR_CONST_STD: return None
    return (a - mu) / sd

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from statsmodels.tsa.api import VAR
    except Exception as e:
        print(f"[ERROR] statsmodels required: {e}")
        return

    sig = _load_signal()
    if sig is None or sig.empty: return

    rows = []
    for ind, ticker in SECTOR_TO_ETF.items():
        edir = OUT_DIR / _slug(ind)
        edir.mkdir(parents=True, exist_ok=True)

        sec = _read_etf(ticker)
        if sec is None:
            print(f"[WARN] Missing ETF for {ind} ({ticker}).")
            continue

        s = sig[sig["industry"] == ind][["date", "sent_mean"]].dropna()
        if s.empty:
            print(f"[INFO] No sentiment for {ind}.")
            continue

        dfm = sec.merge(s, on="date", how="inner").dropna().sort_values("date").reset_index(drop=True)
        if len(dfm) < MIN_OBS:
            print(f"[INFO] Not enough data for VAR ({ind}). Need >= {MIN_OBS}, got {len(dfm)}.")
            continue

        z_ret = _standardize(dfm["ret"])
        z_sent = _standardize(dfm["sent_mean"])
        if z_ret is None or z_sent is None:
            print(f"[INFO] Skipping {ind}: series near-constant after alignment.")
            continue

        X = pd.DataFrame({"ret": z_ret, "sent": z_sent}).dropna().reset_index(drop=True)
        if len(X) < MIN_OBS:
            print(f"[INFO] Not enough clean rows for VAR ({ind}).")
            continue

        X.index = pd.RangeIndex(start=0, stop=len(X))
        auto_maxlag = max(1, min(MAX_LAG_CAP, len(X) // 15))

        model = VAR(X).fit(maxlags=auto_maxlag, ic='aic')
        p = model.k_ar

        # --- FIX: Handle case where AIC selects 0 lags ---
        if p > 0:
            p_sent_to_ret = model.test_causality("ret", ["sent"], kind="f").pvalue
            p_ret_to_sent = model.test_causality("sent", ["ret"], kind="f").pvalue
            print(f"[OK] Granger test for '{ind}' complete with lag order p={p}.")
        else:
            # If lag is 0, causality cannot be tested. P-values are non-significant by definition.
            p_sent_to_ret = 1.0
            p_ret_to_sent = 1.0
            print(f"[INFO] AIC selected 0 lags for '{ind}'. Granger causality test skipped.")
        # --- End FIX ---

        summ = {"industry": ind, "ticker": ticker, "lag_order": p, "n_obs": int(model.nobs),
                "p_sent_to_ret": float(p_sent_to_ret), "p_ret_to_sent": float(p_ret_to_sent)}
        (edir / "var_summary.json").write_text(json.dumps(summ, indent=2))
        rows.append(summ)

    if rows:
        pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False)
        print(f"[DONE] Granger/VAR summary -> {OUT_DIR/'summary.csv'}")
    else:
        print("[WARN] No VAR results written.")

if __name__ == "__main__":
    main()