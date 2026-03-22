# xai-finnews-sentiment

Explainable financial news sentiment toolkit — end‑to‑end, scriptable, and reproducible.

This repo ships **interpretable baselines**, **FinBERT**, **multilingual inference**, **weak labels**, and **econometrics** (event study, VAR/Granger, backtests) for financial news — with every figure/table produced by a CLI script under `scripts/`.

---

## Features at a glance

- **Interpretable baselines**
  - TF‑IDF + Logistic Regression with **SHAP** exports
  - **Glass‑box EBM (GAM)** baseline (via `interpret`)
- **Transformer models**
  - **FinBERT** (ProsusAI) zero‑shot inference
  - **Multilingual** sentiment: XLM‑R (CardiffNLP) with nlptown fallback
- **Lexicon baselines**
  - **Loughran–McDonald (LM)** wordlists (rebuild from the official Excel or use included lists)
  - **VADER** (optional) for weak labels/benchmarks
- **Marketaux ingestion**
  - JSON/JSONL → tidy CSVs (articles & article×symbol pairs), language detection, industry mapping
- **Weak supervision**
  - VADER→LM weak labels for Marketaux; per‑industry and overall LR models with SHAP
- **Gold evaluation**
  - Evaluate LM/VADER/LR/FinBERT/WeakLR on your manual labels; auto‑plots confusion matrices
- **Temporal / regime analysis**
  - SHAP **regime shift** (early vs. late; low‑vol vs. high‑vol via sector ETFs)
- **Econometrics**
  - **Event study** (AR/CAR) around high/low sentiment days (market vs mean‑adjusted model)
  - **VAR/Granger** causality tests (does sentiment Granger‑cause returns?)
  - **Backtests** for simple sector ETF strategies using z‑scored sentiment
- **Data hygiene**
  - **Leakage / near‑duplicate guard** (gold ↔ Marketaux; Marketaux ↔ FPB/FiQA; internal dupes)
- **Quality‑of‑life**
  - One‑shot **environment check** / folder scaffold
  - Helper to download **SPDR sector ETFs** from Yahoo Finance

---

## Folder layout


data/
  annotation/                    # manual gold labels (+ batches/ for sampling sheets)
    batches/
  lexicons/
    lm/                          # LM lists & sources; cleaned lists saved here
  market/etf/                    # (optional) sector ETF CSVs via yfinance
  processed/
    marketaux/                   # standardized CSVs used by scripts
  raw/marketaux/                 # place Marketaux export here (news.json / .jsonl)
models/
  weaklr/                        # weak-label LR model checkpoints
outputs/
  baselines/
    finbert/                     # FinBERT metrics/preds
    lr_shap/                     # LR+SHAP metrics/preds/shap_tops
    ebm_gam/                     # EBM metrics/feature importance
    lexicon_metrics.csv
  multilingual/                  # XLM‑R predictions and summaries
  weaklabel_lr/                  # weak‑label LR (overall/by_industry) + daily signal
  gold_eval/                     # gold‑set metrics/preds/confmats
    confmats/
  figures/                       # rendered figures (e.g., confusion matrices)
  regime/                        # regime‑shift SHAP outputs & summary
  event_study/                   # AR/CAR tables & plots per industry
  granger/                       # VAR/Granger summaries
  backtest/                      # strategy daily curves & summaries
  leakage/                       # leakage audit reports
scripts/                          # all CLI pipelines (see below)
Loughran-McDonald_MasterDictionary_1993-2024.xlsx  # (optional, for LM rebuild)
README.md


---

## Installation

Python **3.10+** recommended.

bash
pip install -U   pandas numpy scikit-learn nltk shap matplotlib   datasets transformers accelerate "torch==2.*" sentencepiece   yfinance statsmodels joblib   langdetect langid   interpret
# (optional, if NLTK prompts for it)
python -m nltk.downloader vader_lexicon


> **GPU:** If you plan to use a GPU with PyTorch, install a CUDA‑matched `torch` wheel per the PyTorch docs.

---

## Quickstart (scripted)

> All scripts assume repo root as CWD. Paths are centralized in `scripts/_config.py`.

### 0) Setup & checklist
bash
python scripts/00_setup.py

Creates folders, checks for key files, and writes `outputs/00_setup_status.json` with readiness flags.

### 1) Public benchmarks (FPB & FiQA‑2018 headlines)
bash
python scripts/01_fetch_benchmarks.py

Writes:
- `data/processed/fpb.csv` (text,y)
- `data/processed/fiqa_headlines.csv` (FiQA continuous scores binned via ±0.05 into 3 classes)
- `outputs/dataset_stats.csv`

### 2) Ingest your Marketaux export
Place your export at `data/raw/marketaux/news.json` (JSON array or JSONL), then:
bash
python scripts/02_ingest_marketaux.py

Writes:
- `data/processed/marketaux/marketaux_news_articles.csv` (1 row/article; lang + dominant industry)
- `data/processed/marketaux/marketaux_news_pairs.csv` (article × symbol)

### 2b) Create balanced annotation batches (optional)
bash
python scripts/02b_make_annotation_batches.py

Produces `data/annotation/batches/batch_*.csv` (industry‑balanced, ≥20% non‑EN).

### 3) Build LM wordlists (from CSV/TXT or official Excel)
bash
python scripts/03_build_lm_lists.py

Outputs cleaned lists under `data/lexicons/lm/lm_{category}.txt` and a build report in `outputs/baselines/`.

### 4) Lexicon baselines on FPB & FiQA (+ optional VADER)
bash
python scripts/04_lexicon_baselines.py

Writes metrics to `outputs/baselines/lexicon_metrics.csv` and per‑dataset predictions under `outputs/baselines/`.

### 5) Interpretable baseline: LR (TF‑IDF) + SHAP
bash
python scripts/05_lr_shap_benchmarks.py

Artifacts per dataset in `outputs/baselines/lr_shap/` + saved model under `models/`.

### 5b) Glass‑box EBM (GAM‑like) baseline (optional)
bash
python scripts/05b_ebm_gam_benchmarks.py


### 6) FinBERT (ProsusAI) inference
bash
python scripts/06_finbert_benchmarks.py


### 7) Weak‑label LR on Marketaux (EN)
Creates VADER→LM weak labels, trains overall + per‑industry LR, and exports SHAP.
bash
python scripts/07_marketaux_weaklabel_lr.py

Also writes a daily EN sentiment signal per industry to:

outputs/weaklabel_lr/daily_sentiment_by_industry.csv


### 7b) Multilingual sentiment (non‑EN)
bash
python scripts/07b_xlm_infer_marketaux.py


### 8) Evaluate on manual gold annotations
bash
python scripts/08_eval_on_manual_annotations.py

Writes global/per‑industry/per‑language metrics & predictions to `outputs/gold_eval/`.

### 9) Plot confusion matrices (from gold eval)
bash
python scripts/09_plot_confmats.py

Saves per‑model PNGs under `outputs/figures/`.

### 10) SHAP regime‑shift analysis (time & volatility)
bash
python scripts/10_shap_regime_shift_all.py

- Early vs Late (median split by date)
- Low‑vol vs High‑vol (requires sector ETF CSVs; see helper below)

### 10b) Weak‑label vs Gold **bias audit**
bash
python scripts/10_weak_label_bias_audit.py


### 11) **Event study** (AR/CAR) around sentiment events
bash
python scripts/11_event_study.py

Uses the daily signal from step 7. If missing, auto‑builds from Marketaux.

### 12) **VAR/Granger** causality tests
bash
python scripts/12_granger_var.py


### 13) **Backtests**: z‑score strategies on sector ETFs
bash
python scripts/13_backtest_sector_strategies.py


### 15) Leakage / near‑duplicate audit
bash
python scripts/15_data_leakage_guard.py


### Helper: fetch SPDR sector ETFs (+ SPY) from Yahoo
bash
python scripts/_helper_fetch_sector_etfs.py

Writes CSVs to `data/market/etf/` with standardized OHLCV columns.

---

## Key outputs (examples)

- `*_metrics.json`, `*_metrics.csv` — model metrics (acc, macro‑F1, per‑class F1)
- `*_predictions.csv` — text, y/y_pred (+ calibrated probabilities where available)
- `gold_eval/confmats/*.json` → `figures/confmat_*.png` — confusion matrices
- `lr_shap/*_shap_top_tokens.csv` — token‑level explainability (per class)
- `regime/*/time_top_tokens_*.csv` and `vol_top_tokens_*.csv` — regime drivers
- `event_study/*/car_table.csv` — AR/CAR by side/window (+ quick plots)
- `granger/summary.csv` — VAR lag selection & Granger p‑values
- `backtest/*/summary_*.json` — trading metrics, daily equity curves
- `weaklabel_lr/daily_sentiment_by_industry.csv` — daily signal used by econometrics
- `leakage/*.csv` — overlap/duplicate reports + `exclude_article_ids.txt`

---

## Data sources & downloads

- **Benchmarks:** FPB (`takala/financial_phrasebank`, `sentences_allagree`) and FiQA‑2018 (`TheFinAI/fiqa-sentiment-classification`) fetched by `01_fetch_benchmarks.py` (via Hugging Face).
- **Models:** `ProsusAI/finbert` (Transformers). XLM‑R: `cardiffnlp/twitter-xlm-roberta-base-sentiment` (fallback `nlptown/bert-base-multilingual-uncased-sentiment`).
- **VADER:** downloaded automatically by NLTK if missing.
- **LM lexicons:** cleaned lists under `data/lexicons/lm/`. Rebuild from the official **Loughran–McDonald Master Dictionary (1993–2024)** using `03_build_lm_lists.py`.
- **Marketaux news:** **not** fetched by the repo. Place your export at `data/raw/marketaux/news.json`.
- **Sector ETFs:** optional, via `scripts/_helper_fetch_sector_etfs.py` (uses `yfinance`).

---

## Reproducibility notes

- Deterministic seeds where applicable (`SEED=42` in `_config.py`); minor variance possible across library versions.
- CPU works for everything except heavy Transformer inference; GPU recommended for Transformers.
- Paths, thresholds, industry sets, and batch sizes are centralized in `scripts/_config.py`.

---

## Licensing

- **Code (this repo):** MIT License  
- **Manual labels** (`data/annotation/annotated_articles.csv`) and docs: CC BY 4.0  
- **Third‑party assets:** original licenses/terms apply  
- **Loughran–McDonald Master Dictionary:** downloaded from Notre Dame SRAF — check their terms before redistribution  
- **Marketaux content:** subject to Marketaux API Terms  
- **ProsusAI/FinBERT, FPB, FiQA, VADER, XLM‑R:** governed by original licenses

---

## Citation

If this repo helps your work, please cite:

bibtex
@software{XAI_FinNews_Sentiment_2025,
  title  = {xai-finnews-sentiment: Explainable Financial News Sentiment Toolkit},
  year   = {2025},
  author = {Repository Authors},
  url    = {https://github.com/MaraAlexandru/xai-finnews-sentiment}
}


---

## Acknowledgements

- Loughran & McDonald for the Master Dictionary  
- ProsusAI for FinBERT  
- CardiffNLP & nlptown for multilingual models  
- FPB and FiQA dataset contributors  
- Marketaux for API access
