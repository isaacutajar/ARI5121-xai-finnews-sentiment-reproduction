# scripts/02_ingest_marketaux.py
"""
Ingests Marketaux-style news JSON → tidy CSVs with language + industries.

Inputs (hardcoded):
  data/raw/marketaux/news.json  # JSON array or JSONL

Outputs:
  data/processed/marketaux/marketaux_news_articles.csv  # 1 row/article
  data/processed/marketaux/marketaux_news_pairs.csv     # 1 row/article×symbol

Run: python scripts/02_ingest_marketaux.py
"""

import json, hashlib, re
from pathlib import Path
from datetime import datetime
import pandas as pd

from _config import (
    RAW_MARKETAUX_JSON, PROCESSED_MARKETAUX_DIR,
    MARKETAUX_ARTICLES_CSV, MARKETAUX_PAIRS_CSV,
    INDUSTRIES, DROP_INDICES_ETFS_FX, TARGET_LANGS, SEED
)

# --------- Language detection (robust w/ fallbacks) ---------
def detect_lang(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "unk"
    # Try langdetect
    try:
        from langdetect import detect as _ld  # pip install langdetect
        lang = _ld(t)
        return lang
    except Exception:
        pass
    # Try langid
    try:
        import langid  # pip install langid
        return langid.classify(t)[0]
    except Exception:
        pass
    # Simple heuristic fallback: ASCII → en-ish; presence of specific diacritics → latin EU
    if re.search(r"[À-ÿ]", t):
        # naive: pick likely romance language token hints
        if re.search(r"\b(el|la|los|las|de|y|en|con|para|por)\b", t.lower()):
            return "es"
        if re.search(r"\b(le|la|les|des|et|en|pour|par)\b", t.lower()):
            return "fr"
        if re.search(r"\b(il|la|gli|le|di|e|per|con)\b", t.lower()):
            return "it"
        if re.search(r"[åäö]", t.lower()):
            return "sv"
        if re.search(r"[øæå]", t.lower()):
            return "da"
        if re.search(r"[ñ]", t.lower()):
            return "es"
        return "eu"
    return "en"  # default

# --------- Helpers ---------
FX = {"USD","EUR","JPY","GBP","CHF","CAD","AUD","NZD","CNY","HKD","SGD","SEK","NOK","DKK","ZAR","MXN","BRL","INR"}
ETF_HINTS = {"ETF","EXCHANGE TRADED FUND","FUND","ISHARES","VANGUARD","SPDR","SELECT SECTOR"}
INDEX_HINTS = {"INDEX","INDICES","DOW","S&P","SECTOR","COMPOSITE"}

def is_index_etf_fx(sym: str, name: str) -> bool:
    s = (sym or "").upper()
    n = (name or "").upper()
    if s.startswith("^"): return True
    if "/" in s: return True
    if len(s) == 6 and s[:3] in FX and s[3:] in FX: return True
    if any(k in n for k in ETF_HINTS): return True
    if any(k in n for k in INDEX_HINTS): return True
    return False

def read_json_any(path: Path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.read(1); f.seek(0)
        if first == "[":
            data = json.load(f)
            return data if isinstance(data, list) else [data]
        rows = []
        for line in f:
            line = line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except Exception: pass
        return rows

def mk_id(url: str, title: str, uuids):
    if isinstance(uuids, list) and uuids:
        return str(uuids[0])
    h = hashlib.sha1(f"{title}|{url}".encode("utf-8", errors="ignore")).hexdigest()
    return h[:12]

def norm_ts(ts):
    if not ts: return ""
    s = str(ts).replace("Z","+00:00")
    try:
        return datetime.fromisoformat(s).isoformat()
    except Exception:
        try:
            return pd.to_datetime(s, errors="coerce", utc=True).isoformat()
        except Exception:
            return ""

def norm_list(values):
    out, seen = [], set()
    for v in values or []:
        v = (v or "").strip()
        if not v or v in seen: continue
        seen.add(v); out.append(v)
    return out

def main():
    PROCESSED_MARKETAUX_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_MARKETAUX_JSON.exists():
        print(f"[ERROR] Missing input: {RAW_MARKETAUX_JSON}")
        return

    raw = read_json_any(RAW_MARKETAUX_JSON)
    if not raw:
        print(f"[WARN] 0 records in {RAW_MARKETAUX_JSON}")
        return

    seen_urls = set()
    art_rows, pair_rows = [], []

    for r in raw:
        title = (r.get("title") or "").strip()
        desc  = (r.get("description") or "").strip()
        url   = (r.get("url") or "").strip()
        if not url:  # skip if no URL anchor
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)

        aid = mk_id(url, title, r.get("uuids"))
        published_at = norm_ts(r.get("published_at"))
        source = (r.get("source") or "").strip()

        syms = r.get("symbols") or []
        tickers, companies, inds = [], [], []
        for s in syms:
            sym = (s.get("symbol") or "").strip()
            comp= (s.get("company_name") or "").strip()
            ind = (s.get("industry") or "").strip()
            if DROP_INDICES_ETFS_FX and is_index_etf_fx(sym, comp):
                continue
            if sym or comp or ind:
                tickers.append(sym)
                companies.append(comp)
                inds.append(ind)
                pair_rows.append({
                    "article_id": aid,
                    "published_at": published_at,
                    "source": source,
                    "url": url,
                    "title": title,
                    "symbol": sym,
                    "company_name": comp,
                    "industry": ind
                })

        inds = [i for i in inds if i]
        dom_ind = inds[0] if inds else ""
        # clean text for lang detection
        joined = " ".join([x for x in [title, desc] if x]).strip()
        lang = detect_lang(joined)

        art_rows.append({
            "article_id": aid,
            "published_at": published_at,
            "source": source,
            "url": url,
            "title": title,
            "description": desc,
            "text": (f"{title} — {desc}" if title and desc else (title or desc)),
            "tickers": ", ".join(norm_list(tickers)),
            "industries": ", ".join(norm_list(inds)),
            "dominant_industry": dom_ind,
            "language": lang
        })

    df_articles = pd.DataFrame(art_rows)
    df_pairs    = pd.DataFrame(pair_rows)
    # Light post-processing
    if not df_articles.empty:
        df_articles = df_articles.sort_values(["published_at","source"], ascending=[False, True])

    df_articles.to_csv(MARKETAUX_ARTICLES_CSV, index=False, encoding="utf-8-sig")
    df_pairs.to_csv(MARKETAUX_PAIRS_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] Articles: {len(df_articles)} → {MARKETAUX_ARTICLES_CSV}")
    print(f"[OK] Pairs   : {len(df_pairs)} → {MARKETAUX_PAIRS_CSV}")

    # Quick sector/lang counts
    by_ind = (
        df_articles.assign(dominant_industry=df_articles["dominant_industry"].fillna(""))
                   .groupby("dominant_industry").size().sort_values(ascending=False)
    )
    by_lang = df_articles.groupby("language").size().sort_values(ascending=False)
    print("\nTop industries (dominant):")
    print(by_ind.head(15).to_string())
    print("\nLanguages:")
    print(by_lang.to_string())

if __name__ == "__main__":
    main()
