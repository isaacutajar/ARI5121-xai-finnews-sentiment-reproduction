# scripts/11_event_study.py
"""
Event Study: abnormal returns (AR/CAR) around high/low daily sentiment events.

Uses the high-quality sentiment signal from the fine-tuned FinBERT model.

If data/market/etf/SPY.csv exists -> Market Model:
  r_sector_t = alpha + beta * r_spy_t + eps_t  (estimated in [-120,-21])
Else -> Mean-adjusted model:
  AR_t = r_sector_t - mean(r_sector_t in [-120,-21])

Events: daily sentiment z-score (63d rolling) > +1.5 (POS) or < -1.5 (NEG).
Windows: estimation [-120,-21], event windows [-1,+3] and [-5,+5].

Inputs:
- outputs/finetune_eval/daily_sentiment_finetuned.csv (Primary Signal)
- data/market/etf/{XLC,XLY,XLE,XLF,XLV,XLI,XLRE,XLU,XLK}.csv (Sector ETFs)
- data/market/etf/SPY.csv (Market Proxy)

Outputs (per industry):
- outputs/event_study/<slug>/events.csv
- outputs/event_study/<slug>/car_table.csv
- outputs/event_study/<slug>/car_plot_<win>.png
Global:
- outputs/event_study/summary.csv

Run: python scripts/11_event_study.py
"""
from pathlib import Path
import re
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

RNG = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ETF_DIR = DATA_DIR / "market" / "etf"
OUT_DIR = OUTPUTS_DIR / "event_study"
# CORRECTED: Point to the new high-quality signal file
SIG_CSV = OUTPUTS_DIR / "finetune_eval" / "daily_sentiment_finetuned.csv"
SPY_PATH = ETF_DIR / "SPY.csv"

SECTOR_TO_ETF = {
    "Communication Services": "XLC", "Consumer Cyclical": "XLY", "Energy": "XLE",
    "Financial Services": "XLF", "Healthcare": "XLV", "Industrials": "XLI",
    "Real Estate": "XLRE", "Utilities": "XLU", "Technology": "XLK",
}

def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", (s or "unknown").lower()).strip("-")

def _read_etf(ticker: str):
    p = ETF_DIR / f"{ticker}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["Date"])
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
    df["date"] = df["Date"].dt.date
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

def _event_windows():
    return {"[-1,+3]": (-1, 3), "[-5,+5]": (-5, 5)}

def _compute_ar_car(sec_ret, mkt_ret, event_date, est_win=(-120, -21), use_market_model=True):
    dates = sec_ret["date"].tolist()
    if event_date not in set(dates): return None
    idx = dates.index(event_date)
    est_start, est_end = idx + est_win[0], idx + est_win[1]
    if est_start < 0 or est_end <= est_start: return None
    
    est = sec_ret.iloc[est_start:est_end+1].copy()
    if len(est) < 30: return None

    if use_market_model and mkt_ret is not None:
        est = est.merge(mkt_ret, on="date", how="inner", suffixes=("", "_mkt")).dropna()
        if len(est) < 30: return None
        X = est["ret_mkt"].values
        Y = est["ret"].values
        X = np.vstack([np.ones_like(X), X]).T
        try:
            alpha, beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        except Exception:
            return None
    else:
        alpha, beta = est["ret"].mean(), 0.0

    out = {}
    for name, (start_offset, end_offset) in _event_windows().items():
        lb, ub = idx + start_offset, idx + end_offset
        if lb < 0 or ub >= len(sec_ret):
            out[name] = None
            continue
        window = sec_ret.iloc[lb:ub+1].copy()
        if use_market_model and mkt_ret is not None:
            window = window.merge(mkt_ret, on="date", how="inner", suffixes=("", "_mkt")).dropna(subset=["ret", "ret_mkt"])
            if window.empty:
                out[name] = None
                continue
            exp = alpha + beta * window["ret_mkt"].values
        else:
            exp = np.full(len(window), alpha)
        ar = window["ret"].values - exp
        out[name] = {"dates": window["date"].astype(str).tolist(), "ar": ar.tolist(), "car": np.cumsum(ar).tolist()}
    return out

def _tstat_mean(x):
    x = np.array(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2: return None
    m, s = x.mean(), x.std(ddof=1)
    return None if s == 0 else float(m / (s / np.sqrt(len(x))))

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sig = _load_signal()
    if sig is None or sig.empty: return

    res = []
    spy = _read_etf("SPY") if SPY_PATH.exists() else None
    use_market_model = spy is not None

    for ind, ticker in SECTOR_TO_ETF.items():
        edir = OUT_DIR / _slug(ind)
        edir.mkdir(parents=True, exist_ok=True)

        sec = _read_etf(ticker)
        if sec is None:
            print(f"[WARN] Missing or unreadable ETF CSV for {ind} ({ticker}) at {ETF_DIR}.")
            continue

        s = sig[sig["industry"] == ind].dropna(subset=["sent_mean"]).copy()
        if s.empty:
            print(f"[INFO] No sentiment for {ind}.")
            continue

        s = s.sort_values("date").reset_index(drop=True)
        s["z"] = _zscore_roll(pd.Series(s["sent_mean"].values, index=pd.to_datetime(s["date"])), win=63, minp=40).values
        pos_ev = s[s["z"] >= 1.5]["date"].tolist()
        neg_ev = s[s["z"] <= -1.5]["date"].tolist()
        pd.DataFrame({"side": ["POS"] * len(pos_ev) + ["NEG"] * len(neg_ev), "date": pos_ev + neg_ev}).to_csv(edir / "events.csv", index=False)

        car_tables = []
        for side, ev_dates in {"POS": pos_ev, "NEG": neg_ev}.items():
            cars = {name: [] for name in _event_windows().keys()}
            for d in ev_dates:
                arcar = _compute_ar_car(sec, spy, d, use_market_model=use_market_model)
                if arcar:
                    for name, obj in arcar.items():
                        if obj and obj.get("car"):
                            cars[name].append(obj["car"][-1])
            
            for name, vals in cars.items():
                vals = [v for v in vals if v is not None and np.isfinite(v)]
                n = len(vals)
                car_tables.append({
                    "industry": ind, "ticker": ticker, "model": "Market" if use_market_model else "MeanAdj",
                    "side": side, "window": name, "n_events": n,
                    "car_mean": float(np.mean(vals)) if n else None,
                    "tstat": _tstat_mean(vals) if n else None
                })
        
        car_df = pd.DataFrame(car_tables)
        car_df.to_csv(edir / "car_table.csv", index=False)
        res.extend(car_tables)

    if res:
        pd.DataFrame(res).to_csv(OUT_DIR / "summary.csv", index=False)
        print(f"[DONE] Event study summary -> {OUT_DIR/'summary.csv'}")
    else:
        print(f"[WARN] No results written. Check signals and ETF CSVs under: {ETF_DIR}")

if __name__ == "__main__":
    main()