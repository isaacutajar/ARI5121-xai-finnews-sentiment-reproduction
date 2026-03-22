# scripts/02b_make_annotation_batches.py
"""
Creates balanced annotation batches (industry coverage + ≥20% non-EN).

Reads:
  data/processed/marketaux/marketaux_news_articles.csv
Avoids overlap with:
  data/annotation/annotated_articles.csv   (if present)

Writes:
  data/annotation/batches/batch_01.csv, batch_02.csv, ...

Each row has: article_id, published_at, source, url, industry, language, title,
              description, text, y (blank), rationale (blank), scope (blank)
The user must manually fill y as numbers: 0=negative, 1=neutral, 2=positive.
Run: python scripts/02b_make_annotation_batches.py
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd

from _config import (
    MARKETAUX_ARTICLES_CSV, ANNOTATED_GOLD_CSV,
    ANNOTATION_BATCHES_DIR, INDUSTRIES, SEED,
    ANNOTATION_TARGET_TOTAL, ANNOTATION_BATCH_SIZE,
    NON_EN_MIN_SHARE, TARGET_LANGS
)

RNG = np.random.default_rng(SEED)

def _safe_str(x): return "" if pd.isna(x) else str(x)

def main():
    ANNOTATION_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    if not MARKETAUX_ARTICLES_CSV.exists():
        print(f"[ERROR] Missing: {MARKETAUX_ARTICLES_CSV}")
        return

    df = pd.read_csv(MARKETAUX_ARTICLES_CSV)

    # exclude already annotated gold set if present
    exclude_ids = set()
    if ANNOTATED_GOLD_CSV.exists():
        try:
            dgold = pd.read_csv(ANNOTATED_GOLD_CSV)
            if "article_id" in dgold.columns:
                exclude_ids = set(dgold["article_id"].astype(str))
                print(f"Leakage guard: excluding {len(exclude_ids)} already-gold articles")
        except Exception:
            pass
    if "article_id" in df.columns:
        df = df[~df["article_id"].astype(str).isin(exclude_ids)]

    # only rows with some text
    df["text"] = df["text"].fillna("")
    df = df[df["text"].str.len() >= 15].copy()
    if df.empty:
        print("[ERROR] No eligible rows after filtering.")
        return

    # language normalization
    df["language"] = df["language"].fillna("unk").str.lower()
    df.loc[~df["language"].isin(TARGET_LANGS + ["unk"]), "language"] = "other"

    # dominant industry normalization
    df["dominant_industry"] = df["dominant_industry"].fillna("")
    # Keep all industries in scope, but mark empty as "Unknown"
    df.loc[df["dominant_industry"].eq(""), "dominant_industry"] = "Unknown"

    target_total = min(ANNOTATION_TARGET_TOTAL, len(df))
    n_batches = max(1, math.ceil(target_total / ANNOTATION_BATCH_SIZE))
    per_batch = ANNOTATION_BATCH_SIZE

    # We’ll stratify each batch by industry and language (≥20% non-en overall).
    # Build pools
    df_en  = df[df["language"].eq("en")]
    df_non = df[~df["language"].eq("en")]

    # Compute desired non-EN share per batch (at least NON_EN_MIN_SHARE)
    desired_non_en = max(NON_EN_MIN_SHARE, 0.20)

    # Industry list with Unknown appended if present
    inds = list(INDUSTRIES)
    if "Unknown" in df["dominant_industry"].unique():
        inds = inds + ["Unknown"]
    inds_unique = [i for i in inds if i in set(df["dominant_industry"].unique())]

    batches_written = 0
    used_ids = set()

    for b in range(1, n_batches + 1):
        batch_need = min(per_batch, target_total - batches_written * per_batch)
        if batch_need <= 0:
            break

        # quota per language
        q_non = int(round(batch_need * desired_non_en))
        q_en  = batch_need - q_non

        # sample by industry within language buckets ~ proportional to availability
        take_rows = []

        def sample_bucket(bucket_df, q, label):
            if q <= 0 or bucket_df.empty:
                return pd.DataFrame(columns=bucket_df.columns)
            # proportional by industry present in this bucket
            sizes = bucket_df["dominant_industry"].value_counts()
            weights = (sizes / sizes.sum()).to_dict()
            allo = {i: max(1, int(round(q * weights.get(i, 0)))) for i in inds_unique}
            # fix rounding
            total_allo = sum(allo.values())
            while total_allo > q:
                # decrement largest allocations first
                i = max(allo, key=lambda k: allo[k])
                allo[i] = max(0, allo[i] - 1); total_allo -= 1
            while total_allo < q:
                # increment smallest non-zero
                i = min(allo, key=lambda k: allo[k])
                allo[i] += 1; total_allo += 1

            picks = []
            for ind, k in allo.items():
                if k <= 0: continue
                pool = bucket_df[(bucket_df["dominant_industry"] == ind) & (~bucket_df["article_id"].astype(str).isin(used_ids))]
                if pool.empty: continue
                k = min(k, len(pool))
                picks.append(pool.sample(n=k, random_state=SEED))
            if picks:
                out = pd.concat(picks, ignore_index=True)
                out["__lang_bucket__"] = label
                return out
            return pd.DataFrame(columns=bucket_df.columns)

        pick_non = sample_bucket(df_non, q_non, "non-en")
        pick_en  = sample_bucket(df_en,  q_en,  "en")

        batch_df = pd.concat([pick_non, pick_en], ignore_index=True)
        # If short, top up from remaining pool
        if len(batch_df) < batch_need:
            pool = df[~df["article_id"].astype(str).isin(used_ids.union(set(batch_df["article_id"].astype(str))))]
            k = min(batch_need - len(batch_df), len(pool))
            if k > 0:
                batch_df = pd.concat([batch_df, pool.sample(n=k, random_state=SEED)], ignore_index=True)

        if batch_df.empty:
            break

        # clip to exact batch_need and mark used
        if len(batch_df) > batch_need:
            batch_df = batch_df.sample(n=batch_need, random_state=SEED)
        used_ids.update(batch_df["article_id"].astype(str))

        # prepare annotation sheet
        out = batch_df[[
            "article_id","published_at","source","url","dominant_industry","language","title","description","text"
        ]].copy()
        out.rename(columns={"dominant_industry":"industry"}, inplace=True)
        out.insert(len(out.columns), "y", "")          # 0/1/2 to be filled by annotators
        out.insert(len(out.columns), "rationale", "")  # free-text
        out.insert(len(out.columns), "scope", "")      # company/industry scope if needed

        # write
        fname = ANNOTATION_BATCHES_DIR / f"batch_{b:02d}.csv"
        out.to_csv(fname, index=False, encoding="utf-8-sig")
        batches_written += 1
        print(f"[OK] Wrote {len(out):4d} rows → {fname}")

    total_selected = len(used_ids)
    print(f"\nTarget total: {target_total} | Selected unique articles: {total_selected}")
    print(f"Batches written: {batches_written} × {ANNOTATION_BATCH_SIZE} (last may be smaller)")

if __name__ == "__main__":
    main()
