# scripts/15_data_leakage_guard.py
"""
Leakage & Near-Duplicate Guard (robust to label-only gold CSV)

What it checks
- GOLD vs. Marketaux: exact URL/title overlaps + near-duplicates (64-bit simhash)
- Benchmarks (FPB, FiQA) vs. Marketaux: near-duplicates
- Duplicates within Marketaux (by URL and headline/text normalization)

Key robustness
- Gold may contain only: article_id,y,rationale,scope
  → we enrich from Marketaux by article_id (title/description/text/url/industry/language)
- Stray header rows in gold are removed
- All text columns are optional; safe fallbacks are used

Inputs (if present)
  data/processed/marketaux/marketaux_news_articles.csv
  data/annotation/annotated_articles.csv
  data/processed/fpb.csv
  data/processed/fiqa_headlines.csv

Outputs
  outputs/leakage/leakage_report.json
  outputs/leakage/overlaps_exact_url_title.csv
  outputs/leakage/overlaps_simhash_gold_vs_marketaux.csv
  outputs/leakage/overlaps_benchmarks_similar.csv
  outputs/leakage/exclude_article_ids.txt
  outputs/leakage/dupes_within_corpus.csv

Run: python scripts/15_data_leakage_guard.py
"""

from pathlib import Path
import json, re, hashlib
import pandas as pd
import numpy as np

from _config import (
    MARKETAUX_ARTICLES_CSV, ANNOTATED_GOLD_CSV, FPB_CSV, FIQA_HEADLINES_CSV,
    OUTPUTS_DIR
)

OUT_DIR = OUTPUTS_DIR / "leakage"
SIMHASH_BITS = 64
SIMHASH_HAMMING_THR = 3   # <=3 bits difference → near-duplicate
MIN_CANON_LEN = 15        # skip very short strings for similarity

# ---------- Utilities ----------

def _read_csv(path: Path, parse_dates=None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype=str, parse_dates=parse_dates, on_bad_lines="skip")
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None

def _is_uuidish(s: str) -> bool:
    if not isinstance(s, str): return False
    s = s.strip().lower()
    if s in {"article_id", ""}: return False
    return bool(re.fullmatch(r"[0-9a-f-]{12,36}", s))

def _norm_text(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[“”\"'`´]", "", t)
    t = re.sub(r"[\u200b-\u200f]", "", t)
    return t.strip()

def _shingles(tokens, k=4):
    if len(tokens) < k:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i:i+k]) for i in range(len(tokens)-k+1)]

def _simhash64(text_norm: str, k=4) -> int:
    if not text_norm:
        return 0
    toks = re.findall(r"[a-z0-9]+", text_norm)
    sh = _shingles(toks, k=k)
    if not sh:
        return 0
    v = np.zeros(SIMHASH_BITS, dtype=int)
    for s in sh:
        h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
        for i in range(SIMHASH_BITS):
            v[i] += 1 if ((h >> i) & 1) else -1
    out = 0
    for i in range(SIMHASH_BITS):
        if v[i] >= 0:
            out |= (1 << i)
    return out

def _hamming64(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))

def _mk_fallback_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df

def _prepare_text_fields(df: pd.DataFrame,
                         title_col="title", desc_col="description", text_col="text") -> pd.DataFrame:
    """
    Creates canonical string for similarity:
      __title__, __desc__, __text__, __canon__, __canon_norm__, __simhash__
    Canon = title if present/lengthy else text else (title + ' — ' + desc).
    """
    df = df.copy()
    # add fallbacks as empty series of right length
    if title_col not in df.columns:
        df[title_col] = ""
    if desc_col and desc_col not in df.columns:
        df[desc_col] = ""
    if text_col not in df.columns:
        df[text_col] = ""

    df["__title__"] = df[title_col].fillna("").astype(str)
    df["__desc__"] = df[desc_col].fillna("").astype(str) if desc_col else ""
    df["__text__"] = df[text_col].fillna("").astype(str)

    # build missing text from title/description if needed
    needs_text = df["__text__"].eq("")
    df.loc[needs_text, "__text__"] = (df.loc[needs_text, "__title__"] + " — " + df.loc[needs_text, "__desc__"]).str.strip(" —")

    # choose canonical string
    # prefer title if reasonably long; else text
    title_long = df["__title__"].str.len() >= 8
    df["__canon__"] = np.where(title_long & df["__title__"].ne(""),
                               df["__title__"], df["__text__"])
    df["__canon_norm__"] = df["__canon__"].map(_norm_text)
    df["__simhash__"] = df["__canon_norm__"].map(_simhash64)
    return df

def _pairs_similar(a_df: pd.DataFrame, b_df: pd.DataFrame,
                   a_name: str, b_name: str, hthr: int = SIMHASH_HAMMING_THR) -> pd.DataFrame:
    """
    Finds pairs with Hamming distance <= hthr using simhash of __canon_norm__.
    Lightweight blocking by first character of normalized canon.
    """
    a = a_df[a_df["__canon_norm__"].str.len() >= MIN_CANON_LEN].copy()
    b = b_df[b_df["__canon_norm__"].str.len() >= MIN_CANON_LEN].copy()
    if a.empty or b.empty:
        return pd.DataFrame(columns=["a_id","b_id","a_title","b_title","hamming","a_src","b_src"])

    a["__key__"] = a["__canon_norm__"].str[:1]
    b["__key__"] = b["__canon_norm__"].str[:1]
    bmap = {k: sub for k, sub in b.groupby("__key__")}

    rows = []
    for _, ra in a.iterrows():
        cand = bmap.get(ra["__key__"])
        if cand is None or cand.empty:
            continue
        ha = int(ra["__simhash__"])
        # vectorized dist
        dists = cand["__simhash__"].apply(lambda hb: _hamming64(ha, int(hb)))
        idx = dists[dists <= hthr].index.tolist()
        for j in idx:
            rb = cand.loc[j]
            rows.append({
                "a_id": ra.get("article_id", ""),
                "b_id": rb.get("article_id", ""),
                "a_title": ra["__canon__"],
                "b_title": rb["__canon__"],
                "hamming": int(_hamming64(ha, int(rb["__simhash__"]))),
                "a_src": a_name, "b_src": b_name
            })
    return pd.DataFrame(rows)

# ---------- Main ----------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_ma = _read_csv(MARKETAUX_ARTICLES_CSV, parse_dates=["published_at"])
    df_gold_raw = _read_csv(ANNOTATED_GOLD_CSV)
    df_fpb = _read_csv(FPB_CSV)
    df_fiqa = _read_csv(FIQA_HEADLINES_CSV)

    report = {
        "present": {
            "marketaux": df_ma is not None,
            "gold": df_gold_raw is not None,
            "fpb": df_fpb is not None,
            "fiqa": df_fiqa is not None,
        },
        "summary": {}
    }

    if df_ma is None or df_ma.empty:
        print(f"[ERROR] Missing or empty {MARKETAUX_ARTICLES_CSV}. Run 02_ingest_marketaux.py first.")
        (OUT_DIR / "leakage_report.json").write_text(json.dumps(report, indent=2))
        return

    # Normalize Marketaux
    df_ma = df_ma.fillna("")
    if "article_id" not in df_ma.columns:
        # fallback id by URL (stable)
        df_ma["article_id"] = df_ma["url"].astype(str).map(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest()[:12])
    # Prepare text fields for similarity
    df_ma_prep = _prepare_text_fields(df_ma, title_col="title", desc_col="description", text_col="text")

    # Duplicates within Marketaux (by URL & by normalized canon)
    dup_url = df_ma_prep[df_ma_prep["url"].ne("") & df_ma_prep["url"].duplicated(keep=False)]
    dup_canon = df_ma_prep[df_ma_prep["__canon_norm__"].duplicated(keep=False)]
    if not dup_url.empty or not dup_canon.empty:
        dupe_out = pd.concat([
            dup_url.assign(dupe_key="url")[["article_id","url","__canon__","published_at","dupe_key"]],
            dup_canon.assign(dupe_key="canon")[["article_id","url","__canon__","published_at","dupe_key"]],
        ], ignore_index=True).drop_duplicates()
        dupe_out.to_csv(OUT_DIR / "dupes_within_corpus.csv", index=False)

    exclude_ids = set()

    # ---------- GOLD vs Marketaux ----------
    if df_gold_raw is not None and not df_gold_raw.empty:
        # sanitize gold (remove stray header rows)
        df_gold_raw = df_gold_raw.fillna("")
        if "article_id" in df_gold_raw.columns:
            df_gold_raw = df_gold_raw[df_gold_raw["article_id"].astype(str).map(_is_uuidish)]
            df_gold_raw = df_gold_raw.drop_duplicates(subset=["article_id"], keep="last")
        else:
            df_gold_raw["article_id"] = ""

        # enrich gold from Marketaux by article_id (title/desc/text/url/etc.)
        gold_cols = ["article_id","y","rationale","scope"]
        for c in gold_cols:
            if c not in df_gold_raw.columns:
                df_gold_raw[c] = ""
        keep_from_ma = ["article_id","published_at","source","url","dominant_industry","language","title","description","text"]
        for c in keep_from_ma:
            if c not in df_ma_prep.columns:
                df_ma_prep[c] = ""
        df_gold = df_gold_raw.merge(df_ma_prep[keep_from_ma], on="article_id", how="left", validate="m:1")
        df_gold = df_gold.rename(columns={"dominant_industry":"industry"}).fillna("")

        # prepare gold for similarity (works even if title/text still blank)
        df_gold_prep = _prepare_text_fields(df_gold, title_col="title", desc_col="description", text_col="text")

        # exact URL overlap
        url_overlap = set(df_gold_prep["url"].astype(str)) & set(df_ma_prep["url"].astype(str))
        url_overlap.discard("")  # drop empties

        # exact title/text (canon) overlap
        title_overlap = set(df_gold_prep["__canon_norm__"]) & set(df_ma_prep["__canon_norm__"])

        exact_rows = []
        if url_overlap:
            exact_rows.append(
                pd.merge(
                    df_gold_prep[df_gold_prep["url"].isin(url_overlap)][["article_id","url","__canon__"]].rename(columns={"article_id":"gold_article_id","__canon__":"gold_canon"}),
                    df_ma_prep[df_ma_prep["url"].isin(url_overlap)][["article_id","url","__canon__"]].rename(columns={"article_id":"ma_article_id","__canon__":"ma_canon"}),
                    on="url", how="inner"
                )
            )
        if title_overlap:
            exact_rows.append(
                pd.merge(
                    df_gold_prep[df_gold_prep["__canon_norm__"].isin(title_overlap)][["article_id","__canon__","__canon_norm__"]].rename(columns={"article_id":"gold_article_id","__canon__":"gold_canon"}),
                    df_ma_prep[df_ma_prep["__canon_norm__"].isin(title_overlap)][["article_id","__canon__","__canon_norm__"]].rename(columns={"article_id":"ma_article_id","__canon__":"ma_canon"}),
                    on="__canon_norm__", how="inner"
                )
            )
        if exact_rows:
            pd.concat(exact_rows, ignore_index=True).drop_duplicates().to_csv(
                OUT_DIR / "overlaps_exact_url_title.csv", index=False
            )

        # simhash near-duplicates
        near = _pairs_similar(df_gold_prep, df_ma_prep, "GOLD", "MARKETAUX", hthr=SIMHASH_HAMMING_THR)
        if not near.empty:
            near.to_csv(OUT_DIR / "overlaps_simhash_gold_vs_marketaux.csv", index=False)

        # exclusion list: ALL gold article_ids
        if "article_id" in df_gold_raw.columns:
            exclude_ids = set(df_gold_raw["article_id"].astype(str))
            (OUT_DIR / "exclude_article_ids.txt").write_text("\n".join(sorted(i for i in exclude_ids if i)))

        report["summary"].update({
            "gold_rows": int(len(df_gold_raw)),
            "gold_enriched_rows": int(len(df_gold)),
            "gold_exact_url": int(len(url_overlap)),
            "gold_exact_title_or_text": int(len(title_overlap)),
            "gold_near_dupe_pairs": int(len(near)),
            "exclude_ids_n": len(exclude_ids),
        })
    else:
        report["summary"]["note"] = "Gold annotations not present; only within-Marketaux duplicate checks performed."

    # ---------- Benchmarks (headline similarity) ----------
    bench_pairs = []
    if df_fpb is not None and not df_fpb.empty:
        # FPB has columns text,y — map text→title
        a = df_fpb.rename(columns={"text":"title"}).copy()
        a = _mk_fallback_cols(a, ["article_id"])  # not present; keep blank
        a = _prepare_text_fields(a, title_col="title", desc_col=None, text_col="title")
        bench_pairs.append(("FPB", a))
    if df_fiqa is not None and not df_fiqa.empty:
        a = df_fiqa.rename(columns={"text":"title"}).copy()
        a = _mk_fallback_cols(a, ["article_id"])
        a = _prepare_text_fields(a, title_col="title", desc_col=None, text_col="title")
        bench_pairs.append(("FiQA", a))

    bench_matches = []
    for name, bdf in bench_pairs:
        near = _pairs_similar(bdf, df_ma_prep, name, "MARKETAUX", hthr=SIMHASH_HAMMING_THR)
        if not near.empty:
            near["bench"] = name
            bench_matches.append(near)
    if bench_matches:
        bb = pd.concat(bench_matches, ignore_index=True)
        bb.to_csv(OUT_DIR / "overlaps_benchmarks_similar.csv", index=False)
        report["summary"]["bench_similar_pairs"] = int(len(bb))

    # ---------- Write report ----------
    (OUT_DIR / "leakage_report.json").write_text(json.dumps(report, indent=2))
    print(f"[DONE] Leakage report → {OUT_DIR/'leakage_report.json'}")
    if exclude_ids:
        print(f"[HINT] Exclusion IDs written → {OUT_DIR/'exclude_article_ids.txt'}")

if __name__ == "__main__":
    main()
