# scripts/16_deep_explain_finbert.py
"""
Deep XAI for FinBERT:
 - Integrated Gradients (Captum) over embeddings
 - LIME (text) local explanations
 - Attention Rollout token importances
 - Faithfulness curves (deleting top-k tokens)

Inputs (priority for samples):
  data/annotation/annotated_articles.csv  (if present, sampled from here; will merge text by article_id)
  else: data/processed/marketaux/marketaux_news_articles.csv  (EN only)

Outputs:
  outputs/deep_xai/examples.csv                 # texts + predictions
  outputs/deep_xai/ig/attributions.jsonl        # per-example token attributions
  outputs/deep_xai/lime/attributions.jsonl
  outputs/deep_xai/attn/attributions.jsonl
  outputs/deep_xai/faithfulness_curves.csv      # method, frac_removed, prob
  outputs/deep_xai/example_cards.md             # human-friendly snippets

Run:
  python 16_deep_explain_finbert.py                # light preset (default)
  python 16_deep_explain_finbert.py --preset heavy # heavy/original-style
"""
from pathlib import Path
import argparse, json, os, re, random
import numpy as np
import pandas as pd
from tqdm import tqdm

from _config import (
    MARKETAUX_ARTICLES_CSV, ANNOTATED_GOLD_CSV, OUTPUTS_DIR, SEED
)

# ---------------------------
# Presets
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["light", "heavy"], default="light",
                    help="Speed/quality preset. 'light' (default) vs 'heavy' (original-style).")
    return ap.parse_args()

args = parse_args()

# default knobs (light)
PRESET = {
    "light": dict(
        N_EXAMPLES=80,
        MAX_TOKENS=96,
        BATCH_SIZE=32,
        IG_STEPS=8,
        LIME_NUM_SAMPLES=1000,
        LIME_NUM_FEATURES=10,
        FAITH_FRACS=[0, 0.1, 0.3, 0.5],
    ),
    "heavy": dict(
        N_EXAMPLES=200,
        MAX_TOKENS=160,
        BATCH_SIZE=16,
        IG_STEPS=20,
        LIME_NUM_SAMPLES=5000,
        LIME_NUM_FEATURES=15,
        FAITH_FRACS=[0, 0.05, 0.1, 0.2, 0.3, 0.5],
    ),
}[args.preset]

# Optional CPU speed knobs (esp. Windows)
os.environ.setdefault("OMP_NUM_THREADS", "4")
try:
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
except Exception:
    pass

random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = OUTPUTS_DIR / "deep_xai"
OUT_DIR_IG = OUT_DIR / "ig"
OUT_DIR_LIME = OUT_DIR / "lime"
OUT_DIR_ATTN = OUT_DIR / "attn"

MODEL_NAME = "ProsusAI/finbert"  # we’ll use config.id2label dynamically

# ---------------------------
# Data loading
# ---------------------------
def _load_samples(n_max, min_len=15):
    """Prefer gold annotations; if they lack text, merge with Marketaux by article_id."""
    gold_p = Path(ANNOTATED_GOLD_CSV)
    mkt_p = Path(MARKETAUX_ARTICLES_CSV)
    df = None

    if gold_p.exists():
        df = pd.read_csv(gold_p)
        text_cols = [c for c in ["text", "title", "description"] if c in df.columns]
        if not text_cols and "article_id" in df.columns and mkt_p.exists():
            m = pd.read_csv(mkt_p)[["article_id", "text", "title", "description", "language"]]
            df = df.merge(m, on="article_id", how="left")
            text_cols = [c for c in ["text", "title", "description"] if c in df.columns]
        if text_cols:
            df["text"] = (
                df[text_cols].fillna("").astype(str)
                .apply(lambda r: " — ".join([z for z in r.values.tolist() if z.strip() and z.strip().lower() != "nan"]), axis=1)
            )
        else:
            print("[WARN] Gold file has no text/title/description, and no Marketaux merge available. Falling back to Marketaux.")
            df = None

        if df is not None and "language" in df.columns:
            df["language"] = df["language"].fillna("unk").str.lower()
            df = df[df["language"].isin(["en", "unk"])].copy()

    if df is None or df.empty:
        if not mkt_p.exists():
            print("[ERROR] No samples available (neither gold with text nor marketaux file exists).")
            return None
        df = pd.read_csv(mkt_p)
        df["language"] = df["language"].fillna("unk").str.lower()
        df = df[df["language"].eq("en")].copy()
        df["text"] = df["text"].fillna("").astype(str)

    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.len() >= min_len].copy()
    if df.empty:
        print("[ERROR] No eligible rows after filtering.")
        return None

    if len(df) > n_max:
        df = df.sample(n_max, random_state=SEED).reset_index(drop=True)

    # Trim extremely long strings to avoid super-long runs
    df["text"] = df["text"].str.slice(0, 2000)

    return df[["text"]].reset_index(drop=True)

# ---------------------------
# Model + prediction
# ---------------------------
def _prep_hf():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            attn_implementation="eager",  # required for output_attentions=True in transformers>=4.36
        )
        model.eval()
        id2label = getattr(model.config, "id2label", {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
        label2id = getattr(model.config, "label2id", {v: k for k, v in id2label.items()})
        # device info (print once)
        try:
            import torch
            device_idx = 0 if torch.cuda.is_available() else -1
            print(f"[INFO] Using device: {'cuda:0' if device_idx==0 else 'cpu'}  |  preset={args.preset}")
        except Exception:
            device_idx = -1
        return tok, model, id2label, label2id, device_idx
    except Exception as e:
        print(f"[ERROR] transformers model load failed: {e}")
        return None, None, None, None, -1

_PIPE = None
def _get_pipe(tok, model, device_idx):
    """Reuse one pipeline instance for speed."""
    global _PIPE
    if _PIPE is None:
        from transformers import TextClassificationPipeline
        _PIPE = TextClassificationPipeline(
            model=model,
            tokenizer=tok,
            truncation=True,
            top_k=None,    # replaces deprecated return_all_scores=True
            device=device_idx
        )
    return _PIPE

def _predict_proba(texts, tok, model, device_idx, batch_size):
    pipe = _get_pipe(tok, model, device_idx)
    arrs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predict", unit="batch"):
        out = pipe(texts[i:i+batch_size])  # list of list[{'label','score'}]
        for lst in out:
            arrs.append([d["score"] for d in lst])
    return np.array(arrs)

# ---------------------------
# Explanations
# ---------------------------
def _integrated_gradients(texts, tok, model, id2label, max_tokens, ig_steps):
    """Integrated Gradients over input embeddings (Captum)."""
    try:
        from captum.attr import IntegratedGradients
        import torch
    except Exception as e:
        print(f"[WARN] Captum not available: {e}")
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def forward_emb(inputs_embeds, attention_mask):
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs.logits

    ig = IntegratedGradients(forward_emb)
    results = []

    for text in tqdm(texts, desc="Integrated Gradients", unit="ex"):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        with __import__("torch").no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            probs = __import__("torch").softmax(logits, dim=-1).detach().cpu().numpy()[0]
            pred_cls = int(__import__("torch").argmax(logits, dim=-1).item())

            emb_layer = model.get_input_embeddings()
            emb = emb_layer(input_ids)  # (1, T, H)

        baseline = __import__("torch").zeros_like(emb)

        attributions, _ = ig.attribute(
            inputs=emb,
            baselines=baseline,
            additional_forward_args=(attn,),
            target=pred_cls,
            n_steps=ig_steps,
            return_convergence_delta=True
        )
        attn_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        toks = tok.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        s = attn_scores
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)

        results.append({
            "text": text,
            "tokens": toks,
            "attr": s.tolist(),
            "pred_idx": pred_cls,
            "probs": probs.tolist()
        })
    return results

def _lime_explain(texts, tok, model, id2label, device_idx, max_tokens, num_samples, num_features, probs_cache):
    try:
        from lime.lime_text import LimeTextExplainer
    except Exception as e:
        print(f"[WARN] LIME not available: {e}")
        return []

    def predict_proba(strs):
        return _predict_proba(strs, tok, model, device_idx, batch_size=PRESET["BATCH_SIZE"])

    # Provide class names in index order 0..K-? if available
    class_names = None
    if isinstance(id2label, dict):
        try:
            class_names = [id2label[i] for i in sorted(id2label.keys())]
        except Exception:
            class_names = None

    explainer = LimeTextExplainer(class_names=class_names)

    results = []
    for idx, text in enumerate(tqdm(texts, desc="LIME", unit="ex")):
        exp = explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=1  # explain the predicted top-1 label
        )
        label_used = exp.available_labels()[0]

        # reuse cached probs if available
        probs = probs_cache.get(idx)
        if probs is None:
            probs = predict_proba([text])[0]
            probs_cache[idx] = probs
        pred = int(np.argmax(probs))

        # map weights to tokens (word-level)
        weights = dict(exp.as_list(label=label_used))
        toks = re.findall(r"\w+|\W", text)[:max_tokens]
        scores = [weights.get(t, 0.0) for t in toks]
        s = np.array(scores, dtype=float)
        if len(s):
            s = (s - s.min()) / (s.max() - s.min() + 1e-8)

        results.append({
            "text": text,
            "tokens": toks,
            "attr": s.tolist(),
            "pred_idx": pred,
            "probs": probs.tolist()
        })
    return results

def _attention_rollout(texts, tok, model, max_tokens):
    import torch
    results = []
    for text in tqdm(texts, desc="Attention Rollout", unit="ex"):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens, add_special_tokens=True)
        with torch.no_grad():
            out = model(**enc, output_attentions=True)
            logits = out.logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            pred = int(torch.argmax(logits, dim=-1).item())
            atts = out.attentions  # tuple of L tensors [B,H,T,T]

        if not atts:
            print("[WARN] output_attentions returned None — skipping this example.")
            continue

        # rollout: avg heads per layer, add residual, row-normalize, multiply through layers
        A = None
        for layer_att in atts:
            a = layer_att.mean(dim=1).squeeze(0)  # (T,T)
            eye = torch.eye(a.size(-1), device=a.device, dtype=a.dtype)
            a = a + eye
            a = a / a.sum(dim=-1, keepdim=True)
            A = a if A is None else A @ a

        # importance to [CLS] (token 0): contributions from tokens → CLS
        imp = A[:, 0].detach().cpu().numpy()
        imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)
        toks = tok.convert_ids_to_tokens(enc["input_ids"].squeeze(0).tolist())

        results.append({
            "text": text,
            "tokens": toks,
            "attr": imp.tolist(),
            "pred_idx": pred,
            "probs": probs.tolist()
        })
    return results

# ---------------------------
# Faithfulness (deletion)
# ---------------------------
def _detok_from_wordpieces(tokens):
    """Very simple WordPiece detokenizer for deletion tests."""
    out = []
    for t in tokens:
        if t in ("[CLS]", "[SEP]", "[PAD]", "[UNK]"):
            continue
        if t.startswith("##") and out:
            out[-1] = out[-1] + t[2:]
        else:
            out.append(t)
    return out

def _faithfulness_curves(method_name, examples, tok, model, id2label, device_idx, fracs):
    from transformers import TextClassificationPipeline
    try:
        pipe = TextClassificationPipeline(model=model, tokenizer=tok, truncation=True, top_k=None, device=device_idx)
    except Exception:
        return pd.DataFrame(columns=["method","frac_removed","prob","example_idx"])

    rows = []
    for i, ex in enumerate(tqdm(examples, desc=f"Faithfulness ({method_name})", unit="ex")):
        toks_wp = ex["tokens"]
        attr = np.array(ex["attr"], dtype=float)
        pred_idx = int(ex["pred_idx"])

        # map to rough words for readable deletion; align lengths if mismatch
        toks = _detok_from_wordpieces(toks_wp)
        if len(toks) < 12:
            continue

        # simple attribution downsampling to word level (mean over contiguous pieces)
        if len(attr) == len(toks_wp):
            word_attr = []
            acc = 0.0; count = 0
            for t, a in zip(toks_wp, attr):
                if t in ("[CLS]", "[SEP]", "[PAD]", "[UNK]"):
                    continue
                if t.startswith("##"):
                    acc += a; count += 1
                else:
                    if count > 0:
                        word_attr.append(acc / max(1, count))
                    acc = a; count = 1
            if count > 0:
                word_attr.append(acc / max(1, count))
            wa = np.array(word_attr, dtype=float)
            if len(wa) != len(toks):
                wa = np.interp(np.linspace(0, len(wa)-1, num=len(toks)), np.arange(len(wa)), wa)
        else:
            wa = np.ones(len(toks), dtype=float)

        order = np.argsort(wa)[::-1]  # high → low
        for frac in fracs:
            k = max(1, int(round(frac * len(toks))))
            keep_mask = np.ones(len(toks), dtype=bool)
            keep_mask[order[:k]] = False
            pruned_text = " ".join([t for t, keep in zip(toks, keep_mask) if keep])
            try:
                out = pipe([pruned_text])[0]  # list[{'label','score'}]
                # find score for the original predicted class
                label_name = id2label.get(pred_idx, str(pred_idx))
                score_map = {d["label"]: d["score"] for d in out}
                sc = score_map.get(label_name, np.nan)
                if np.isnan(sc) and len(out) > pred_idx:
                    sc = out[pred_idx]["score"]
            except Exception:
                sc = np.nan
            rows.append({"method": method_name, "frac_removed": frac, "prob": sc, "example_idx": i})
    return pd.DataFrame(rows)

# ---------------------------
# Main
# ---------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR_IG.mkdir(parents=True, exist_ok=True)
    OUT_DIR_LIME.mkdir(parents=True, exist_ok=True)
    OUT_DIR_ATTN.mkdir(parents=True, exist_ok=True)

    df = _load_samples(PRESET["N_EXAMPLES"])
    if df is None or df.empty:
        return

    tok, model, id2label, label2id, device_idx = _prep_hf()
    if tok is None:
        return

    texts = df["text"].tolist()

    # Predictions table (one pass, cached)
    probs = _predict_proba(texts, tok, model, device_idx, batch_size=PRESET["BATCH_SIZE"])
    preds = probs.argmax(axis=1)
    ex = pd.DataFrame({
        "text": texts,
        "pred_idx": preds,
        "pred_label": [id2label.get(int(i), str(int(i))) for i in preds],
        "p_neg": probs[:,0] if probs.shape[1] > 0 else np.nan,
        "p_neu": probs[:,1] if probs.shape[1] > 1 else np.nan,
        "p_pos": probs[:,2] if probs.shape[1] > 2 else np.nan,
    })
    ex.to_csv(OUT_DIR / "examples.csv", index=False)
    (OUT_DIR / "meta.json").write_text(json.dumps({"model": MODEL_NAME, "id2label": id2label}, indent=2), encoding="utf-8")

    probs_cache = {i: probs[i] for i in range(len(texts))}

    # IG
    ig_res = _integrated_gradients(
        texts, tok, model, id2label,
        max_tokens=PRESET["MAX_TOKENS"],
        ig_steps=PRESET["IG_STEPS"]
    )
    if ig_res:
        with (OUT_DIR_IG / "attributions.jsonl").open("w", encoding="utf-8") as f:
            for item in ig_res:
                f.write(json.dumps(item) + "\n")

    # LIME
    lime_res = _lime_explain(
        texts, tok, model, id2label, device_idx,
        max_tokens=PRESET["MAX_TOKENS"],
        num_samples=PRESET["LIME_NUM_SAMPLES"],
        num_features=PRESET["LIME_NUM_FEATURES"],
        probs_cache=probs_cache
    )
    if lime_res:
        with (OUT_DIR_LIME / "attributions.jsonl").open("w", encoding="utf-8") as f:
            for item in lime_res:
                f.write(json.dumps(item) + "\n")

    # Attention rollout
    attn_res = _attention_rollout(texts, tok, model, max_tokens=PRESET["MAX_TOKENS"])
    if attn_res:
        with (OUT_DIR_ATTN / "attributions.jsonl").open("w", encoding="utf-8") as f:
            for item in attn_res:
                f.write(json.dumps(item) + "\n")

    # Faithfulness curves
    frames = []
    if ig_res:   frames.append(_faithfulness_curves("IG", ig_res, tok, model, id2label, device_idx, PRESET["FAITH_FRACS"]))
    if lime_res: frames.append(_faithfulness_curves("LIME", lime_res, tok, model, id2label, device_idx, PRESET["FAITH_FRACS"]))
    if attn_res: frames.append(_faithfulness_curves("ATTN", attn_res, tok, model, id2label, device_idx, PRESET["FAITH_FRACS"]))
    if frames:
        curves = pd.concat(frames, ignore_index=True)
        curves.to_csv(OUT_DIR / "faithfulness_curves.csv", index=False)

    # Human-friendly markdown cards
    try:
        def _mk_card(items, name):
            lines = [f"## {name} examples\n"]
            for i, exi in enumerate(items[:10]):
                toks = exi["tokens"]; attr = np.array(exi["attr"])
                top = np.argsort(attr)[::-1][:8]
                highlights = [toks[j] for j in top]
                snippet = (exi["text"][:240] + "…").replace("\n", " ")
                lines.append(f"**Example {i+1}**  \nText: {snippet}  \nTop tokens: `{', '.join(highlights)}`  \nPred idx: {exi['pred_idx']}")
            return "\n".join(lines)

        cards = []
        if ig_res:   cards.append(_mk_card(ig_res, "Integrated Gradients"))
        if lime_res: cards.append(_mk_card(lime_res, "LIME"))
        if attn_res: cards.append(_mk_card(attn_res, "Attention Rollout"))
        if cards:
            (OUT_DIR / "example_cards.md").write_text("\n\n---\n\n".join(cards), encoding="utf-8")
    except Exception:
        pass

    print(f"[DONE] Deep XAI artifacts → {OUT_DIR}  (preset={args.preset})")

if __name__ == "__main__":
    main()
