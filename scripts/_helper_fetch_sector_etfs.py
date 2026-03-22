# scripts/_helper_fetch_sector_etfs.py
"""
Downloads daily OHLCV for SPDR sector ETFs + SPY from Yahoo Finance and writes
CSV files to data/market/etf with the exact columns our pipeline expects.

Run (from repo root):
    python scripts/_helper_fetch_sector_etfs.py

Notes:
- No dependency on scripts/_config.py (paths/constants are defined here).
- Uses these tickers: SPY, XLC, XLY, XLE, XLF, XLV, XLI, XLRE, XLU, XLK.
  (We use XLY as a proxy for "Services" in the econometrics scripts.)
"""

from pathlib import Path
import sys
import time

# ---- third-party imports ----
try:
    import pandas as pd
except Exception:
    print("[ERROR] pandas is not installed. Run: pip install pandas")
    sys.exit(1)

try:
    import yfinance as yf  # pip install yfinance
except Exception:
    print("[ERROR] yfinance is not installed. Run: pip install yfinance")
    sys.exit(1)

# ---- paths (standalone, no _config) ----
# Repo root = one level up from this script file
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data" / "market" / "etf"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---- tickers we fetch ----
TICKERS = {
    "SPY": "S&P 500 (market benchmark)",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary (proxy for Services)",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLK": "Technology",
}

# ---- helpers ----
def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the exact column order our pipeline expects:
    Date, Open, High, Low, Close, Adj Close, Volume
    """
    cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    # yfinance returns a DateTimeIndex by default; move it to a column
    df = df.reset_index()

    # Try to ensure a 'Date' column exists
    if "Date" not in df.columns:
        # common alternatives from yfinance
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "Date"})
        else:
            # Fallback: pick the first datetime-like column as Date if present
            dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if dt_cols:
                df = df.rename(columns={dt_cols[0]: "Date"})
            else:
                # create a placeholder; better than failing hard
                df["Date"] = pd.NaT

    # Ensure Adj Close exists; if not provided, fall back to Close
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    # Add any missing required columns
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA

    # Keep only needed columns in exact order
    df = df[cols].copy()

    # Make sure Date is ISO format string (YYYY-MM-DD)
    try:
        df["Date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
    except Exception:
        pass

    return df


def download_one(ticker: str, start="2000-01-01") -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            start=start,
            progress=False,
            auto_adjust=False,
            actions=False,
            threads=True,
        )
    except Exception as e:
        print(f"[ERROR] yfinance failed for {ticker}: {e}")
        return pd.DataFrame()

    return _standardize_ohlcv(df)


def main():
    print(f"Output directory: {OUTDIR}")
    for tkr, sector in TICKERS.items():
        print(f"Downloading {tkr:5s}  ({sector}) ...", end="", flush=True)
        df = download_one(tkr, start="2000-01-01")
        if df.empty or len(df) == 0:
            print(" FAILED (empty)")
            continue
        out = OUTDIR / f"{tkr}.csv"
        try:
            df.to_csv(out, index=False)
            print(f" OK  → {out.name}  (rows: {len(df)})")
        except Exception as e:
            print(f" FAILED (write error: {e})")
        time.sleep(0.5)  # be polite to Yahoo

    print("\nDone. Files are in:", OUTDIR)
    print("Next step: run event/granger/backtest scripts (e.g., scripts\\11_event_study.py).")


if __name__ == "__main__":
    main()
