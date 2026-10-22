#!/usr/bin/env python3
"""
pure_ctc_beam_eval.py

Low-risk A/B test for the CURRENT soup artifact.

What this does
--------------
1. Loads the same MMS model/adapter path used by the current pipeline:
       src/weights/mms-1b-all/adapter.nhi.safetensors (or weights/mms-1b-all/...)
   via model.load_adapter("nhi").
2. Verifies that the active CTC head has the 40-class competition vocabulary.
3. Evaluates the exact same 635 held-out clips used in the previous diagnostics:
       537 nhi clips from tetelancingo_nahuatl/corpus.tsv + audios/
        74 azz clips from nahuatl_dev/metadata.tsv + clips/
        24 nhw clips from nahuatl_dev/metadata.tsv + clips/
4. Produces:
       - current greedy CTC decoding
       - PURE lexicon-free CTC prefix beam search
     with NO KenLM, NO unigrams, NO alpha/beta, NO external LM.
5. Tests several beam widths from the same model forward pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Enable imports from src/ when running from project root
THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR / "src"))
sys.path.append(str(THIS_DIR))

import argparse
import gc
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoProcessor, Wav2Vec2ForCTC

from audio_io import load_audio_16k_mono
from orthography import colonial_to_modern_nahuatl

# ---------------------------------------------------------------------------
# Paths / runtime
# ---------------------------------------------------------------------------

default_weights = (
    THIS_DIR / "src" / "weights" / "mms-1b-all"
    if (THIS_DIR / "src" / "weights" / "mms-1b-all").exists()
    else THIS_DIR / "weights" / "mms-1b-all"
)

MMS_WEIGHTS_DIR = Path(os.environ.get("MMS_WEIGHTS_DIR", default_weights))
SOUP_ADAPTER_PATH = MMS_WEIGHTS_DIR / "adapter.nhi.safetensors"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ---------------------------------------------------------------------------
# Generic file helpers
# ---------------------------------------------------------------------------

AUDIO_COL_CANDIDATES = [
    "audio_filename",
    "filename",
    "audio",
    "audio_file",
    "audio_path",
    "path",
    "file",
    "clip",
    "clip_id",
]

REF_COL_CANDIDATES = [
    "original_sentence",
    "reference",
    "transcript",
    "sentence",
    "text",
    "transcription",
]

LANG_COL_CANDIDATES = [
    "language",
    "lang",
    "dialect",
]


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"TSV not found: {path}")
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        raise ValueError(f"TSV is empty: {path}")
    return df


def detect_column(
    df: pd.DataFrame,
    requested: str | None,
    candidates: Sequence[str],
    purpose: str,
) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(
                f"{purpose} column '{requested}' not found.\n"
                f"Available columns: {list(df.columns)}"
            )
        return requested

    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]

    raise ValueError(
        f"Could not auto-detect {purpose} column.\n"
        f"Tried: {list(candidates)}\n"
        f"Available columns: {list(df.columns)}"
    )


def maybe_detect_language(
    df: pd.DataFrame,
    requested: str | None,
) -> str | None:
    if requested is not None:
        return requested
    lower = {str(c).lower(): c for c in df.columns}
    for cand in LANG_COL_CANDIDATES:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def normalize_space(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def word_error_rate(reference: str, hypothesis: str) -> float:
    r = normalize_space(reference).split()
    h = normalize_space(hypothesis).split()

    if not r:
        return 0.0 if not h else 1.0

    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, start=1):
        cur = [i]
        for j, hw in enumerate(h, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (rw != hw)
            cur.append(min(ins, dele, sub))
        prev = cur

    return prev[-1] / len(r)


def bootstrap_mean_ci(
    deltas: Sequence[float],
    seed: int = 42,
    n_boot: int = 2000,
) -> Tuple[float, float]:
    x = np.asarray(deltas, dtype=np.float64)
    if len(x) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(n_boot, len(x)))
    boot_means = x[indices].mean(axis=1)
    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Dataset loader (Reconstructs exact 537 / 74 / 24 split)
# ---------------------------------------------------------------------------

@dataclass
class ClipRecord:
    dataset: str
    audio_filename: str
    audio_path: Path
    reference: str
    language: str


def resolve_audio(
    value: str,
    audio_dir: Path,
) -> Path:
    raw = Path(str(value))

    candidates = [
        raw,
        audio_dir / raw,
        audio_dir / raw.name,
    ]

    parts = raw.parts
    if parts:
        candidates.append(audio_dir / Path(*parts[-2:]))

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve audio '{value}' under '{audio_dir}'. "
        f"Tried common path interpretations."
    )


def load_records(
    corpus_tsv: Path,
    corpus_audio_dir: Path,
    dev_tsv: Path,
    dev_audio_dir: Path,
    corpus_audio_col: str | None,
    corpus_ref_col: str | None,
    dev_audio_col: str | None,
    dev_ref_col: str | None,
    dev_lang_col: str | None,
) -> List[ClipRecord]:
    # 1. Tetelancingo (nhi) held-out partition -> 537 clips
    corpus = read_tsv(corpus_tsv)
    if "split" in corpus.columns:
        corpus = corpus[corpus["split"] == "train"].reset_index(drop=True)

    corpus_audio = detect_column(corpus, corpus_audio_col, AUDIO_COL_CANDIDATES, "corpus audio")
    corpus_ref = detect_column(corpus, corpus_ref_col, REF_COL_CANDIDATES, "corpus reference")

    speaker_col_corpus = "speaker" if "speaker" in corpus.columns else corpus.columns[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, val_idx = next(gss.split(corpus, groups=corpus[speaker_col_corpus]))
    nhi_val = corpus.iloc[val_idx].reset_index(drop=True)

    # 2. Nahuatl dev (azz & nhw) held-out partition -> 74 azz + 24 nhw
    dev = read_tsv(dev_tsv)
    dev_audio = detect_column(dev, dev_audio_col, AUDIO_COL_CANDIDATES, "dev audio")
    dev_ref = detect_column(dev, dev_ref_col, REF_COL_CANDIDATES, "dev reference")
    dev_lang = maybe_detect_language(dev, dev_lang_col) or "language"
    has_speaker = "speaker" in dev.columns

    def split_dev_dialect(dialect_name: str) -> pd.DataFrame:
        d = dev[dev[dev_lang] == dialect_name].reset_index(drop=True)
        unique_spk = d["speaker"].nunique() if has_speaker else 0
        if has_speaker and unique_spk > 1:
            gss_d = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            _, v_idx = next(gss_d.split(d, groups=d["speaker"]))
            return d.iloc[v_idx].reset_index(drop=True)
        shuffled = d.sample(frac=1.0, random_state=42).reset_index(drop=True)
        return shuffled.iloc[: max(1, int(len(shuffled) * 0.2))].reset_index(drop=True)

    azz_val = split_dev_dialect("azz")
    nhw_val = split_dev_dialect("nhw")

    records: List[ClipRecord] = []

    for row in nhi_val.itertuples(index=False):
        audio_val = str(getattr(row, corpus_audio))
        records.append(
            ClipRecord(
                dataset="nhi",
                audio_filename=audio_val,
                audio_path=resolve_audio(audio_val, corpus_audio_dir),
                reference=str(getattr(row, corpus_ref)),
                language="nhi",
            )
        )

    for df_sub, lang_tag in [(azz_val, "azz"), (nhw_val, "nhw")]:
        for row in df_sub.itertuples(index=False):
            audio_val = str(getattr(row, dev_audio))
            records.append(
                ClipRecord(
                    dataset="nahuatl_dev",
                    audio_filename=audio_val,
                    audio_path=resolve_audio(audio_val, dev_audio_dir),
                    reference=str(getattr(row, dev_ref)),
                    language=lang_tag,
                )
            )

    return records


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def from_pretrained_compat(cls, path: str, **kwargs):
    try:
        return cls.from_pretrained(path, **kwargs)
    except TypeError as e:
        if "dtype" not in kwargs:
            raise
        if "unexpected keyword argument 'dtype'" not in str(e):
            raise
        kwargs = dict(kwargs)
        dtype = kwargs.pop("dtype")
        kwargs["torch_dtype"] = dtype
        return cls.from_pretrained(path, **kwargs)


def load_current_soup():
    if not MMS_WEIGHTS_DIR.exists():
        raise FileNotFoundError(f"MMS directory not found: {MMS_WEIGHTS_DIR}")

    if not SOUP_ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"Expected current soup adapter not found:\n{SOUP_ADAPTER_PATH}"
        )

    size_mb = SOUP_ADAPTER_PATH.stat().st_size / (1024 * 1024)
    print(f"[INFO] adapter.nhi.safetensors size: {size_mb:.2f} MB")
    print(f"[INFO] adapter.nhi.safetensors path: {SOUP_ADAPTER_PATH}")

    model = from_pretrained_compat(
        Wav2Vec2ForCTC,
        str(MMS_WEIGHTS_DIR),
        target_lang="nhi",
        ignore_mismatched_sizes=True,
        local_files_only=True,
        dtype=DTYPE,
    ).to(DEVICE)

    processor = AutoProcessor.from_pretrained(
        str(MMS_WEIGHTS_DIR),
        target_lang="nhi",
        local_files_only=True,
    )

    model.load_adapter("nhi")
    processor.tokenizer.set_target_lang("nhi")
    model.eval()

    vocab_size = len(processor.tokenizer)
    head_size = model.lm_head.out_features

    print(
        f"[INFO] Active tokenizer vocab size: {vocab_size} | "
        f"active lm_head size: {head_size}"
    )

    if vocab_size != 40 or head_size != 40:
        raise RuntimeError(
            "SAFETY CHECK FAILED: active model does not have the expected "
            "40-class competition vocabulary/head. "
            f"Got vocab_size={vocab_size}, head_size={head_size}."
        )

    return model, processor


# ---------------------------------------------------------------------------
# Pure CTC prefix beam search
# ---------------------------------------------------------------------------

NEG_INF = -float("inf")


def logadd(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def token_string_map(processor) -> List[str | None]:
    vocab = processor.tokenizer.get_vocab()
    id_to_token = [None] * len(vocab)
    for tok, idx in vocab.items():
        if 0 <= idx < len(id_to_token):
            id_to_token[idx] = tok

    blank_id = processor.tokenizer.pad_token_id

    symbols: List[str | None] = []
    for idx, tok in enumerate(id_to_token):
        if idx == blank_id:
            symbols.append(None)
        elif tok == "|":
            symbols.append(" ")
        elif tok is None:
            symbols.append(f"<id{idx}>")
        else:
            symbols.append(tok)

    return symbols


def pure_ctc_prefix_beam_search(
    log_probs: np.ndarray,
    blank_id: int,
    symbols: Sequence[str | None],
    beam_width: int = 32,
) -> str:
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")

    T, V = log_probs.shape
    if not (0 <= blank_id < V):
        raise ValueError(f"blank_id={blank_id} outside logits vocabulary V={V}")

    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {
        tuple(): (0.0, NEG_INF)
    }

    token_ids = [
        i for i in range(V)
        if i != blank_id and symbols[i] is not None
    ]

    for t in range(T):
        frame = log_probs[t]

        new_pb: Dict[Tuple[int, ...], float] = {}
        new_pnb: Dict[Tuple[int, ...], float] = {}

        for prefix, (pb, pnb) in beams.items():
            total = logadd(pb, pnb)

            blank_lp = float(frame[blank_id])
            new_pb[prefix] = logadd(
                new_pb.get(prefix, NEG_INF),
                total + blank_lp,
            )

            last = prefix[-1] if prefix else None

            for c in token_ids:
                pc = float(frame[c])

                if c == last:
                    new_pnb[prefix] = logadd(
                        new_pnb.get(prefix, NEG_INF),
                        pnb + pc,
                    )
                    extended = prefix + (c,)
                    new_pnb[extended] = logadd(
                        new_pnb.get(extended, NEG_INF),
                        pb + pc,
                    )
                else:
                    extended = prefix + (c,)
                    new_pnb[extended] = logadd(
                        new_pnb.get(extended, NEG_INF),
                        total + pc,
                    )

        all_prefixes = set(new_pb) | set(new_pnb)
        scored = []
        for prefix in all_prefixes:
            pb2 = new_pb.get(prefix, NEG_INF)
            pnb2 = new_pnb.get(prefix, NEG_INF)
            total2 = logadd(pb2, pnb2)
            scored.append((total2, prefix, pb2, pnb2))

        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:beam_width]

        beams = {
            prefix: (pb2, pnb2)
            for _, prefix, pb2, pnb2 in scored
        }

    best_prefix = max(
        beams.items(),
        key=lambda kv: logadd(kv[1][0], kv[1][1])
    )[0]

    out = "".join(symbols[i] for i in best_prefix if symbols[i] is not None)
    return " ".join(out.split())


def greedy_decode(processor, logits_np: np.ndarray) -> str:
    ids = np.argmax(logits_np, axis=-1)
    return processor.decode(ids, skip_special_tokens=True)


def prepare_log_probs(logits: torch.Tensor) -> np.ndarray:
    x = logits.float()
    x = x - torch.logsumexp(x, dim=-1, keepdim=True)
    return x.cpu().numpy()


@torch.inference_mode()
def infer_logits(
    model,
    processor,
    waveform: np.ndarray,
) -> torch.Tensor:
    inputs = processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
    )
    input_values = inputs.input_values.to(
        device=DEVICE,
        dtype=model.dtype,
    )
    logits = model(input_values).logits[0]

    if logits.shape[-1] != 40:
        raise RuntimeError(
            f"Unexpected logits class dimension: {logits.shape[-1]} (expected 40)."
        )
    return logits


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def run_eval(
    records: List[ClipRecord],
    beam_widths: Sequence[int],
    out_dir: Path,
    limit: int | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if limit is not None:
        records = records[:limit]

    if not records:
        raise RuntimeError("No validation clips available.")

    print(f"[INFO] Evaluating {len(records)} clips.")
    print(f"[INFO] Beam widths: {list(beam_widths)}")
    print("[INFO] Pure decoder: NO LM / NO lexicon / NO KenLM / NO unigrams.")

    model, processor = load_current_soup()
    symbols = token_string_map(processor)
    blank_id = processor.tokenizer.pad_token_id

    print(f"[INFO] tokenizer blank/pad id: {blank_id}")

    rows = []

    for i, rec in enumerate(records, start=1):
        waveform = load_audio_16k_mono(str(rec.audio_path))
        logits = infer_logits(model, processor, waveform)

        logits_np = logits.float().cpu().numpy()
        log_probs = prepare_log_probs(logits)

        greedy_raw = greedy_decode(processor, logits_np)
        greedy_mapped = colonial_to_modern_nahuatl(greedy_raw)

        row = {
            "index": i - 1,
            "dataset": rec.dataset,
            "language": rec.language,
            "audio_filename": rec.audio_filename,
            "reference": rec.reference,
            "greedy_raw": greedy_raw,
            "greedy_mapped": greedy_mapped,
            "greedy_wer": word_error_rate(rec.reference, greedy_mapped),
        }

        for bw in beam_widths:
            beam_raw = pure_ctc_prefix_beam_search(
                log_probs=log_probs,
                blank_id=blank_id,
                symbols=symbols,
                beam_width=bw,
            )
            beam_mapped = colonial_to_modern_nahuatl(beam_raw)

            row[f"beam{bw}_raw"] = beam_raw
            row[f"beam{bw}_mapped"] = beam_mapped
            row[f"beam{bw}_wer"] = word_error_rate(rec.reference, beam_mapped)
            row[f"beam{bw}_same_as_greedy"] = (
                normalize_space(beam_mapped) == normalize_space(greedy_mapped)
            )

        rows.append(row)

        if i % 10 == 0 or i == len(records):
            print(f"[INFO] {i}/{len(records)} clips processed.")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "beam_ab_results.csv", index=False)

    summary = {
        "device": DEVICE,
        "mms_weights_dir": str(MMS_WEIGHTS_DIR),
        "soup_adapter": str(SOUP_ADAPTER_PATH),
        "n_clips": len(df),
        "beam_widths": list(beam_widths),
        "decoder": "pure_ctc_prefix_beam_search",
        "language_model": False,
        "kenlm": False,
        "unigrams": False,
        "greedy_mean_wer": float(df["greedy_wer"].mean()),
    }

    for bw in beam_widths:
        beam_col = f"beam{bw}_wer"
        same_col = f"beam{bw}_same_as_greedy"

        deltas = (
            df[beam_col].to_numpy(dtype=np.float64)
            - df["greedy_wer"].to_numpy(dtype=np.float64)
        )
        ci_lo, ci_hi = bootstrap_mean_ci(deltas)

        summary[f"beam{bw}_mean_wer"] = float(df[beam_col].mean())
        summary[f"beam{bw}_mean_delta_vs_greedy"] = float(deltas.mean())
        summary[f"beam{bw}_paired_bootstrap_95ci"] = [ci_lo, ci_hi]
        summary[f"beam{bw}_identical_to_greedy_fraction"] = float(
            df[same_col].mean()
        )

        print("\n----------------------------------------------------------")
        print(f"Beam width {bw}")
        print(f"  Greedy WER                 : {df['greedy_wer'].mean():.6f}")
        print(f"  Pure beam WER              : {df[beam_col].mean():.6f}")
        print(f"  Delta (beam - greedy)      : {deltas.mean():+.6f}")
        print(f"  95% paired bootstrap CI    : [{ci_lo:+.6f}, {ci_hi:+.6f}]")
        print(f"  Identical to greedy        : {100.0 * df[same_col].mean():.2f}%")

    breakdown = {}
    for lang, g in df.groupby("language", dropna=False):
        lang_key = str(lang)
        breakdown[lang_key] = {
            "n": int(len(g)),
            "greedy_wer": float(g["greedy_wer"].mean()),
        }
        for bw in beam_widths:
            breakdown[lang_key][f"beam{bw}_wer"] = float(g[f"beam{bw}_wer"].mean())

    summary["by_language"] = breakdown

    with open(out_dir / "beam_ab_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==========================================================")
    print("DONE")
    print(f"Results: {out_dir / 'beam_ab_results.csv'}")
    print(f"Summary : {out_dir / 'beam_ab_summary.json'}")
    print("==========================================================")

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Pure lexicon-free CTC beam vs greedy on the current soup."
    )

    ap.add_argument(
        "--corpus-tsv",
        required=True,
        help="e.g. data/tetelancingo_nahuatl/corpus.tsv",
    )
    ap.add_argument(
        "--corpus-audio-dir",
        required=True,
        help="e.g. data/tetelancingo_nahuatl/audios",
    )
    ap.add_argument(
        "--dev-tsv",
        required=True,
        help="e.g. data/nahuatl_dev/metadata.tsv",
    )
    ap.add_argument(
        "--dev-audio-dir",
        required=True,
        help="e.g. data/nahuatl_dev/clips",
    )

    ap.add_argument("--corpus-audio-col", default=None)
    ap.add_argument("--corpus-ref-col", default=None)
    ap.add_argument("--dev-audio-col", default=None)
    ap.add_argument("--dev-ref-col", default=None)
    ap.add_argument("--dev-lang-col", default=None)

    ap.add_argument(
        "--beam-widths",
        nargs="+",
        type=int,
        default=[8, 16, 32],
        help="Default: 8 16 32",
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Quick test limit. Use 10 first; omit for all clips.",
    )

    ap.add_argument(
        "--out-dir",
        default="pure_ctc_beam_output",
    )

    args = ap.parse_args()

    records = load_records(
        corpus_tsv=Path(args.corpus_tsv),
        corpus_audio_dir=Path(args.corpus_audio_dir),
        dev_tsv=Path(args.dev_tsv),
        dev_audio_dir=Path(args.dev_audio_dir),
        corpus_audio_col=args.corpus_audio_col,
        corpus_ref_col=args.corpus_ref_col,
        dev_audio_col=args.dev_audio_col,
        dev_ref_col=args.dev_ref_col,
        dev_lang_col=args.dev_lang_col,
    )

    print(f"[INFO] Loaded nhi records : {sum(r.dataset == 'nhi' for r in records)}")
    print(f"[INFO] Loaded dev records : {sum(r.dataset == 'nahuatl_dev' for r in records)}")
    print(f"[INFO] Total records      : {len(records)}")

    if args.limit is None and len(records) != 635:
        raise RuntimeError(
            f"Expected exactly 635 held-out clips, but loaded {len(records)}.\n"
            "This safety check prevents evaluating the wrong split."
        )

    run_eval(
        records=records,
        beam_widths=args.beam_widths,
        out_dir=Path(args.out_dir),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()