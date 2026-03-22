# scripts/00_setup.py
"""
Creates folders, checks for key files, prints a readiness checklist.

Run: python scripts/00_setup.py
"""

import sys, platform, json, importlib.util
from pathlib import Path
from _config import (
    ROOT, ensure_dirs, RAW_MARKETAUX_JSON, OUTPUTS_DIR, LEXICON_DIR,
    FPB_CSV, FIQA_HEADLINES_CSV, PROCESSED_MARKETAUX_DIR,
    MARKETAUX_ARTICLES_CSV, MARKETAUX_PAIRS_CSV, ANNOTATION_BATCHES_DIR,
    ANNOTATED_GOLD_CSV, INDUSTRIES
)

def main():
    ensure_dirs()
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print("=== Environment ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {pyver}")
    try:
        core_libs = ("numpy", "pandas", "sklearn")
        missing = [lib for lib in core_libs if importlib.util.find_spec(lib) is None]
        if missing:
            print(f"[WARN] Core libs missing: {', '.join(missing)}")
        else:
            print("Core libs: numpy/pandas/sklearn ✓")
    except Exception as e:
        print(f"[WARN] Core libs check error: {e}")

    # folder readiness
    print("\n=== Folder readiness ===")
    for p in [OUTPUTS_DIR, LEXICON_DIR, PROCESSED_MARKETAUX_DIR, ANNOTATION_BATCHES_DIR]:
        print(f"{p.relative_to(ROOT)} ✓ (exists)")

    # checks
    checks = {
        "raw_marketaux_json_exists": RAW_MARKETAUX_JSON.exists(),
        "lm_lexicon_dir_exists": LEXICON_DIR.exists(),
        "gold_annotations_exists": ANNOTATED_GOLD_CSV.exists(),
        "bench_fpb_exists": FPB_CSV.exists(),
        "bench_fiqa_exists": FIQA_HEADLINES_CSV.exists(),
        "marketaux_articles_csv_exists": MARKETAUX_ARTICLES_CSV.exists(),
        "marketaux_pairs_csv_exists": MARKETAUX_PAIRS_CSV.exists(),
    }
    print("\n=== Checklist ===")
    for k, v in checks.items():
        print(f"{k:35s} : {'✓' if v else '✗'}")

    # friendly next-steps
    print("\nNext steps:")
    if not checks["bench_fpb_exists"] or not checks["bench_fiqa_exists"]:
        print("  - Run: python scripts/01_fetch_benchmarks.py")
    if not checks["raw_marketaux_json_exists"]:
        print(f"  - Place your Marketaux news JSON at: {RAW_MARKETAUX_JSON}")
    else:
        if not checks["marketaux_articles_csv_exists"] or not checks["marketaux_pairs_csv_exists"]:
            print("  - Run: python scripts/02_ingest_marketaux.py")
        else:
            print("  - Marketaux CSVs exist ✓")
            if not ANNOTATION_BATCHES_DIR.exists() or not any(ANNOTATION_BATCHES_DIR.glob("batch_*.csv")):
                print("  - Run: python scripts/02b_make_annotation_batches.py")

    # write a machine-readable status (optional)
    status_path = OUTPUTS_DIR / "00_setup_status.json"
    status_path.write_text(json.dumps(checks, indent=2))
    print(f"\nWrote status → {status_path.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
