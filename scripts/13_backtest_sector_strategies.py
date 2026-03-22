# scripts/13_backtest_sector_strategies.py
"""
Backtests: sentiment-tilt strategies on sector ETFs.

Uses the high-quality sentiment signal from the fine-tuned FinBERT model.

Signal:
- Daily mean sentiment per industry.
- Rolling 63-day z-score z_t.
- Position_t =
    Long-Only  : 1 if z_t > +1.0, 0 otherwise
    Long-Short : 1 if z_t > +1.0, -1 if z_t < -1.0, 0 otherwise
- Trades occur at next close: use position_{t-1} * return_t

Costs:
- Per change in position: cost_bps (default 5 bps) * |pos_t - pos_{t-1}|

Outputs (per industry):
- outputs/backtest/<slug>/daily_<variant>.csv
- outputs/backtest/<slug>/summary_<variant>.json
Global:
- outputs/backtest/summary.csv

Run: python scripts/13_backtest_sector_strategies.py
"""
from pathlib import Path
import re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
OUT_DIR = OUTPUTS_DIR / "backtest"
# CORRECTED: Point to the new high-quality signal file
SIG_CSV = OUTPUTS_DIR / "finetune_eval" / "daily_sentiment_finetuned.csv"

COST_BPS = 5
Z_THR = 1.0
MIN_OVERLAP = 25

SECTOR_TO_ETF = {
    "Communication Services": "XLC", "Consumer Cyclical": "XLY", "Energy": "XLE",
    "Financial Services": "XLF", "Healthcare": "XLV", "Industrials": "XLI",
    "Real Estate": "XLRE", "Utilities": "XLU", "Technology": "XLK",
}

def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", (s or "unknown").lower()).strip("-")

def _read_etf(ticker: str):
    p = ETF_DIR / f"{ticker}.csv"
    if not p.exists(): return None
    df = pd.read_csv(p, parse_dates=["Date"])
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["date"] = df["Date"].dt.date
    df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
    df["ret"] = df["Adj Close"].pct_change()
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

def _zscore_roll(x, win=63, minp=40):
    s = x.rolling(win, min_periods=minp)
    return (x - s.mean()) / s.std(ddof=0)

def _perf_stats(r):
    r = np.array(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {"cagr": None, "vol_ann": None, "sharpe": None,
                "max_dd": None, "hit_rate": None, "n_days": 0}
    n = len(r)
    mean_d = r.mean()
    std_d = r.std(ddof=0)
    vol_ann = std_d * np.sqrt(252)
    sharpe = (mean_d / std_d) * np.sqrt(252) if std_d > 0 else 0.0
    eq = (1 + r).cumprod()
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    max_dd = float(dd.min())
    cagr = float(eq[-1]**(252.0 / n) - 1.0) if n > 0 else 0.0
    hit = float((r > 0).mean()) if n > 0 else 0.0
    return {"cagr": cagr, "vol_ann": float(vol_ann), "sharpe": float(sharpe),
            "max_dd": max_dd, "hit_rate": hit, "n_days": int(n)}

def _backtest_variant(dfm, variant="long_only", z_thr=Z_THR, cost_bps=COST_BPS):
    z = _zscore_roll(dfm["sent_mean"])
    z_vals = z.values
    
    pos = np.zeros_like(z_vals, dtype=float)
    pos[z_vals > z_thr] = 1.0
    if variant == "long_short":
        pos[z_vals < -z_thr] = -1.0

    pos_shift = np.insert(pos[:-1], 0, 0.0)
    ret = dfm["ret"].values * pos_shift

    change = np.abs(np.diff(np.insert(pos, 0, 0.0)))
    cost = (cost_bps / 10000.0) * change
    net = ret - cost[:len(ret)]

    out = dfm[["date"]].copy()
    out["pos"] = pos_shift
    out["ret_gross"] = ret
    out["cost"] = cost[:len(ret)]
    out["ret_net"] = net
    out["eq_curve"] = (1 + out["ret_net"]).cumprod()

    stats = _perf_stats(out["ret_net"])
    stats.update({"z_thr": z_thr, "cost_bps": cost_bps, "variant": variant})
    return out, stats

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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

        dfm = sec.merge(s, on="date", how="inner").dropna().sort_values("date")
        if len(dfm) < MIN_OVERLAP:
            print(f"[INFO] Not enough overlap for backtest ({ind}). Need >= {MIN_OVERLAP}, got {len(dfm)}.")
            continue

        for variant in ["long_only", "long_short"]:
            daily, stats = _backtest_variant(dfm, variant=variant, z_thr=Z_THR, cost_bps=COST_BPS)
            daily.to_csv(edir / f"daily_{variant}.csv", index=False)
            (edir / f"summary_{variant}.json").write_text(json.dumps({"industry": ind, "ticker": ticker, **stats}, indent=2))
            rows.append({"industry": ind, "ticker": ticker, "variant": variant, **stats})

    if rows:
        pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False)
        print(f"[DONE] Backtest summary -> {OUT_DIR/'summary.csv'}")
    else:
        print("[WARN] No backtest results written.")

if __name__ == "__main__":
    main()