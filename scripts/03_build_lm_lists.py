# scripts/03_build_lm_lists.py
"""
Build LM wordlists from files placed in data/lexicons/lm/*  → cleaned .txt lists.

Accepted inputs in LEXICON_DIR:
  - CSV with columns like: Word, Positive, Negative, Uncertainty, Litigious, Constraining, Superfluous
  - TXT with one word per line (use subfolder names or filename to infer category)

Outputs:
  data/lexicons/lm/lm_{category}.txt
  outputs/baselines/lm_build_report.json
Run: python scripts/03_build_lm_lists.py
"""
from pathlib import Path
import re, json
import pandas as pd
from _config import LEXICON_DIR, OUTPUTS_DIR

OUT_DIR = LEXICON_DIR  # keep lists alongside source
REPORT_DIR = OUTPUTS_DIR / "baselines"
CATS = ["positive","negative","uncertainty","litigious","constraining","superfluous"]

def _clean_word(w: str) -> str:
    w = (w or "").strip().lower()
    w = re.sub(r"[^a-z]+", " ", w).strip()
    return w

def _unique_sorted(words):
    words = [_clean_word(w) for w in words if _clean_word(w)]
    return sorted(set(words))

def _load_from_csv(path: Path):
    df = pd.read_csv(path)
    # Try common variants
    word_col = None
    for c in df.columns:
        if str(c).strip().lower() in {"word","term","token"}:
            word_col = c; break
    if word_col is None:
        # fallback: first column
        word_col = df.columns[0]
    cols = {c.lower(): c for c in df.columns}
    bags = {k: [] for k in CATS}
    for k in CATS:
        src = cols.get(k)
        if src is None:
            continue
        # binary flags or scores
        for w, flag in zip(df[word_col], df[src]):
            try:
                val = float(flag)
            except Exception:
                val = 1.0 if str(flag).strip() not in {"0","0.0","","nan","NaN"} else 0.0
            if val and str(val) != "0.0":
                bags[k].append(str(w))
    return {k: _unique_sorted(v) for k, v in bags.items()}

def _load_from_txt(path: Path):
    cat = None
    fname = path.stem.lower()
    for k in CATS:
        if k in fname:
            cat = k; break
    if cat is None and path.parent != LEXICON_DIR:
        # infer from parent folder name
        for k in CATS:
            if k in path.parent.name.lower():
                cat = k; break
    words = [l.strip() for l in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    words = _unique_sorted(words)
    if cat:
        return {cat: words}
    # Unknown category: ignore but record in report
    return {"unknown": words}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not LEXICON_DIR.exists():
        print(f"[ERROR] Missing directory: {LEXICON_DIR}")
        return

    bags = {k: [] for k in CATS}
    unknown = []

    any_file = False
    for p in LEXICON_DIR.rglob("*"):
        if p.is_dir(): 
            continue
        if p.suffix.lower() == ".csv":
            any_file = True
            try:
                d = _load_from_csv(p)
                for k, v in d.items():
                    if k in CATS:
                        bags[k].extend(v)
                    else:
                        unknown.extend(v)
                print(f"[OK] parsed CSV: {p.name}")
            except Exception as e:
                print(f"[WARN] failed to parse {p.name}: {e}")
        elif p.suffix.lower() in {".txt",".tsv"}:
            any_file = True
            d = _load_from_txt(p)
            for k, v in d.items():
                if k in CATS:
                    bags[k].extend(v)
                else:
                    unknown.extend(v)

    if not any_file:
        print(f"[WARN] No lexicon files found in {LEXICON_DIR}. Place LM files and re-run.")
        return

    # Deduplicate & save
    summary = {}
    for k in CATS:
        uniq = _unique_sorted(bags[k])
        (OUT_DIR / f"lm_{k}.txt").write_text("\n".join(uniq), encoding="utf-8")
        summary[k] = len(uniq)
    if unknown:
        (OUT_DIR / "lm_unknown.txt").write_text("\n".join(_unique_sorted(unknown)), encoding="utf-8")
        summary["unknown"] = len(_unique_sorted(unknown))

    report = {
        "source_dir": str(LEXICON_DIR),
        "outputs": {k: str((OUT_DIR / f'lm_{k}.txt').relative_to(OUT_DIR.parent.parent)) for k in CATS},
        "counts": summary,
    }
    (REPORT_DIR / "lm_build_report.json").write_text(json.dumps(report, indent=2))
    print("[DONE] LM lists built:", summary)

if __name__ == "__main__":
    main()
