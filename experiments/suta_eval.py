#!/usr/bin/env python3
"""
suta_eval.py

FINAL ACOUSTIC EXPERIMENT:
Single-Utterance Test-Time Adaptation (SUTA) on the EXISTING SOUP.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to module lookup
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, str(THIS_DIR / "src"))

import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoProcessor, Wav2Vec2ForCTC

from audio_io import load_audio_16k_mono
from orthography import colonial_to_modern_nahuatl

# ---------------------------------------------------------------------------
# Dataset record
# ---------------------------------------------------------------------------

@dataclass
class ClipRecord:
    dialect: str
    dataset: str
    audio_filename: str
    audio_path: Path
    reference: str


# ---------------------------------------------------------------------------
# Generic CSV/TSV helpers
# ---------------------------------------------------------------------------

AUDIO_COLS = [
    "audio_filename",
    "filename",
    "audio",
    "audio_file",
    "audio_path",
    "path",
    "file",
    "clip",
]

REF_COLS = [
    "original_sentence",
    "reference",
    "transcript",
    "sentence",
    "text",
    "transcription",
]

LANG_COLS = [
    "language",
    "lang",
    "dialect",
]


def detect_column(
    df: pd.DataFrame,
    requested: str | None,
    candidates: Sequence[str],
    label: str,
) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(f"{label} column '{requested}' not found.")
        return requested

    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    raise ValueError(f"Could not auto-detect {label} column.")


def maybe_detect_column(
    df: pd.DataFrame,
    requested: str | None,
    candidates: Sequence[str],
) -> str | None:
    if requested is not None:
        return detect_column(df, requested, candidates, "language")

    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def resolve_audio(value: str, audio_dir: Path) -> Path:
    raw = Path(str(value))
    candidates = [raw, audio_dir / raw, audio_dir / raw.name]

    if len(raw.parts) >= 2:
        candidates.append(audio_dir / Path(*raw.parts[-2:]))

    seen = set()
    for p in candidates:
        p = p.expanduser()
        if p in seen:
            continue
        seen.add(p)
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(f"Could not resolve audio '{value}' under '{audio_dir}'.")


def load_validation_records(args) -> List[ClipRecord]:
    # 1. Held-out Tetelancingo (nhi) -> 537 clips
    corpus = pd.read_csv(Path(args.corpus_tsv), sep="\t")
    if "split" in corpus.columns:
        corpus = corpus[corpus["split"] == "train"].reset_index(drop=True)

    corpus_audio_col = detect_column(corpus, args.corpus_audio_col, AUDIO_COLS, "corpus audio")
    corpus_ref_col = detect_column(corpus, args.corpus_ref_col, REF_COLS, "corpus reference")

    speaker_col = "speaker" if "speaker" in corpus.columns else corpus.columns[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, val_idx = next(gss.split(corpus, groups=corpus[speaker_col]))
    nhi_val = corpus.iloc[val_idx].reset_index(drop=True)

    # 2. Held-out Nahuatl dev (azz & nhw) -> 74 azz + 24 nhw
    dev = pd.read_csv(Path(args.dev_tsv), sep="\t")
    dev_audio_col = detect_column(dev, args.dev_audio_col, AUDIO_COLS, "dev audio")
    dev_ref_col = detect_column(dev, args.dev_ref_col, REF_COLS, "dev reference")
    dev_lang_col = maybe_detect_column(dev, args.dev_lang_col, LANG_COLS) or "language"
    has_speaker = "speaker" in dev.columns

    def split_dev_dialect(dialect_name: str) -> pd.DataFrame:
        d = dev[dev[dev_lang_col] == dialect_name].reset_index(drop=True)
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
    corpus_audio_dir = Path(args.corpus_audio_dir)
    dev_audio_dir = Path(args.dev_audio_dir)

    for row in nhi_val.itertuples(index=False):
        audio_name = str(getattr(row, corpus_audio_col))
        records.append(
            ClipRecord(
                dialect="nhi",
                dataset="tetelancingo_nahuatl",
                audio_filename=audio_name,
                audio_path=resolve_audio(audio_name, corpus_audio_dir),
                reference=str(getattr(row, corpus_ref_col)),
            )
        )

    for df_sub, lang_tag in [(azz_val, "azz"), (nhw_val, "nhw")]:
        for row in df_sub.itertuples(index=False):
            audio_name = str(getattr(row, dev_audio_col))
            records.append(
                ClipRecord(
                    dialect=lang_tag,
                    dataset="nahuatl_dev",
                    audio_filename=audio_name,
                    audio_path=resolve_audio(audio_name, dev_audio_dir),
                    reference=str(getattr(row, dev_ref_col)),
                )
            )

    return records


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def wer(reference: str, hypothesis: str) -> float:
    r = normalize_text(reference).split()
    h = normalize_text(hypothesis).split()

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


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------

def load_current_production_model():
    weights_dir = (
        THIS_DIR / "src" / "weights" / "mms-1b-all"
        if (THIS_DIR / "src" / "weights" / "mms-1b-all").exists()
        else THIS_DIR / "weights" / "mms-1b-all"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = Wav2Vec2ForCTC.from_pretrained(
        str(weights_dir),
        target_lang="nhi",
        ignore_mismatched_sizes=True,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        str(weights_dir),
        target_lang="nhi",
        local_files_only=True,
    )

    model.load_adapter("nhi")
    processor.tokenizer.set_target_lang("nhi")
    model.eval()

    vocab_size = len(processor.tokenizer)
    head_size = int(model.lm_head.out_features)
    blank_id = int(processor.tokenizer.pad_token_id)

    print(f"[MODEL] vocab={vocab_size} | lm_head={head_size} | blank_id={blank_id}")

    if vocab_size != 40 or head_size != 40:
        raise RuntimeError(
            f"SAFETY CHECK FAILED: expected 40-token competition model, got {vocab_size}/{head_size}."
        )

    return model, processor, device


# ---------------------------------------------------------------------------
# Parameter selection / snapshot
# ---------------------------------------------------------------------------

def configure_trainable_parameters(
    model,
    adapt_head: bool,
) -> List[tuple[str, torch.nn.Parameter]]:
    for p in model.parameters():
        p.requires_grad_(False)

    selected: List[tuple[str, torch.nn.Parameter]] = []
    for name, param in model.named_parameters():
        is_adapter = "adapter_layer" in name
        is_head = adapt_head and "lm_head" in name
        if is_adapter or is_head:
            param.requires_grad_(True)
            selected.append((name, param))

    if not selected:
        raise RuntimeError("No trainable adapter parameters found.")

    trainable_count = sum(p.numel() for _, p in selected)
    print(f"[TRAINABLE] tensors={len(selected)} | parameters={trainable_count:,}")

    if trainable_count > 15_000_000:
        raise RuntimeError(f"SAFETY CHECK FAILED: {trainable_count:,} trainable parameters.")

    if torch.cuda.is_available():
        for _, param in selected:
            if param.dtype != torch.float32:
                param.data = param.data.float()

    return selected


def snapshot_trainable(selected: List[tuple[str, torch.nn.Parameter]]):
    return {name: param.detach().cpu().clone().float() for name, param in selected}


@torch.no_grad()
def restore_trainable(selected: List[tuple[str, torch.nn.Parameter]], snapshot):
    for name, param in selected:
        saved = snapshot[name]
        param.data.copy_(saved.to(device=param.device, dtype=param.dtype))


# ---------------------------------------------------------------------------
# Forward & Decode
# ---------------------------------------------------------------------------

def autocast_context(device: str):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    class _NullContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    return _NullContext()


@torch.inference_mode()
def forward_logits_inference(model, processor, waveform, device: str) -> torch.Tensor:
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    model_input_dtype = next(model.parameters()).dtype
    input_values = inputs.input_values.to(device=device, dtype=model_input_dtype)

    with autocast_context(device):
        logits = model(input_values).logits[0]

    return logits.detach()


def forward_logits_train(model, processor, waveform, device: str) -> torch.Tensor:
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    model_input_dtype = next(model.parameters()).dtype
    input_values = inputs.input_values.to(device=device, dtype=model_input_dtype)

    with autocast_context(device):
        logits = model(input_values).logits[0]

    return logits


def greedy_decode(processor, logits: torch.Tensor) -> str:
    ids = torch.argmax(logits, dim=-1)
    return processor.decode(ids, skip_special_tokens=True)


def baseline_active_mask(baseline_logits: torch.Tensor, blank_id: int) -> torch.Tensor:
    with torch.no_grad():
        ids = torch.argmax(baseline_logits.detach(), dim=-1)
        mask = (ids != blank_id).detach().bool()
    return mask


def entropy_loss_on_active_frames(
    logits: torch.Tensor,
    active_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if active_mask.sum().item() == 0:
        raise RuntimeError("No active non-blank frames available.")

    scaled = logits.float() / float(temperature)
    active_scaled = scaled[active_mask]

    log_probs = torch.log_softmax(active_scaled, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.mean()


def grad_norm(selected: List[tuple[str, torch.nn.Parameter]]) -> float:
    sq = 0.0
    for _, param in selected:
        if param.grad is None:
            continue
        g = param.grad.detach().float()
        if not torch.isfinite(g).all():
            return float("nan")
        sq += float(torch.sum(g * g).item())
    return math.sqrt(sq)


def character_count(text: str) -> int:
    return len("".join(str(text).split()))


# ---------------------------------------------------------------------------
# One-clip adaptation
# ---------------------------------------------------------------------------

def adapt_one_clip(
    model,
    processor,
    waveform,
    baseline_logits,
    baseline_text,
    selected,
    snapshot,
    device,
    lr: float,
    steps: int,
    temperature: float,
    length_guard: float,
):
    blank_id = int(processor.tokenizer.pad_token_id)
    active_mask = baseline_active_mask(baseline_logits, blank_id)
    active_count_before = int(active_mask.sum().item())

    if active_count_before == 0:
        return {
            "status": "baseline_blank_skip",
            "final_text": baseline_text,
            "used_adaptation": False,
            "loss": float("nan"),
            "grad_norm": float("nan"),
            "active_frames_before": 0,
            "active_frames_after": 0,
            "chars_before": character_count(baseline_text),
            "chars_after": character_count(baseline_text),
            "rollback": True,
        }

    optimizer = torch.optim.SGD([p for _, p in selected], lr=lr)
    max_grad_seen = 0.0
    last_loss = float("nan")
    adaptation_failed = False
    failure_reason = ""

    try:
        for step_idx in range(steps):
            optimizer.zero_grad(set_to_none=True)
            logits = forward_logits_train(model, processor, waveform, device)
            loss = entropy_loss_on_active_frames(logits, active_mask, temperature)

            if not torch.isfinite(loss):
                adaptation_failed = True
                failure_reason = "nonfinite_loss"
                break

            loss.backward()
            gnorm = grad_norm(selected)

            if not math.isfinite(gnorm):
                adaptation_failed = True
                failure_reason = "nonfinite_gradient"
                break

            max_grad_seen = max(max_grad_seen, gnorm)
            last_loss = float(loss.detach().float().item())

            if gnorm > 1_000.0:
                adaptation_failed = True
                failure_reason = "gradient_explosion"
                break

            optimizer.step()

    except RuntimeError as e:
        adaptation_failed = True
        failure_reason = f"runtime_error:{type(e).__name__}:{e}"

    if adaptation_failed:
        restore_trainable(selected, snapshot)
        return {
            "status": "rollback",
            "failure_reason": failure_reason,
            "final_text": baseline_text,
            "used_adaptation": False,
            "loss": last_loss,
            "grad_norm": max_grad_seen,
            "active_frames_before": active_count_before,
            "active_frames_after": active_count_before,
            "chars_before": character_count(baseline_text),
            "chars_after": character_count(baseline_text),
            "rollback": True,
        }

    adapted_logits = forward_logits_inference(model, processor, waveform, device)
    adapted_text = greedy_decode(processor, adapted_logits)

    baseline_chars = character_count(baseline_text)
    adapted_chars = character_count(adapted_text)

    active_after = int((torch.argmax(adapted_logits.detach(), dim=-1) != blank_id).sum().item())

    if baseline_chars == 0:
        bad_length = adapted_chars != 0
    else:
        lo = length_guard * baseline_chars
        hi = (2.0 - length_guard) * baseline_chars
        bad_length = not (lo <= adapted_chars <= hi)

    if bad_length:
        restore_trainable(selected, snapshot)
        return {
            "status": "length_guard_rollback",
            "failure_reason": "length_out_of_bounds",
            "final_text": baseline_text,
            "candidate_text": adapted_text,
            "used_adaptation": False,
            "loss": last_loss,
            "grad_norm": max_grad_seen,
            "active_frames_before": active_count_before,
            "active_frames_after": active_after,
            "chars_before": baseline_chars,
            "chars_after": adapted_chars,
            "rollback": True,
        }

    result = {
        "status": "adapted",
        "failure_reason": "",
        "final_text": adapted_text,
        "candidate_text": adapted_text,
        "used_adaptation": True,
        "loss": last_loss,
        "grad_norm": max_grad_seen,
        "active_frames_before": active_count_before,
        "active_frames_after": active_after,
        "chars_before": baseline_chars,
        "chars_after": adapted_chars,
        "rollback": False,
    }

    restore_trainable(selected, snapshot)
    return result


def select_safety_subset(records: List[ClipRecord], seed: int = 42) -> List[ClipRecord]:
    rng = random.Random(seed)
    by_dialect = {}
    for r in records:
        by_dialect.setdefault(r.dialect.lower(), []).append(r)

    selected = []
    requested = {"nhi": 4, "azz": 3, "nhw": 3}

    for dialect, n in requested.items():
        pool = list(by_dialect.get(dialect, []))
        rng.shuffle(pool)
        if len(pool) < n:
            raise RuntimeError(f"Safety test requested {n} '{dialect}' clips but only {len(pool)} found.")
        selected.extend(pool[:n])

    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Evaluation Loop
# ---------------------------------------------------------------------------

def evaluate(records: List[ClipRecord], args):
    model, processor, device = load_current_production_model()
    selected = configure_trainable_parameters(model, adapt_head=args.adapt_head)
    model.eval()
    snapshot = snapshot_trainable(selected)

    print(
        f"[SUTA] lr={args.lr} | steps={args.steps} | "
        f"temperature={args.temperature} | adapt_head={args.adapt_head}"
    )

    rows = []
    for i, rec in enumerate(records, start=1):
        waveform = load_audio_16k_mono(str(rec.audio_path))

        baseline_logits = forward_logits_inference(model, processor, waveform, device)
        baseline_raw = greedy_decode(processor, baseline_logits)
        baseline_text = colonial_to_modern_nahuatl(baseline_raw)
        baseline_wer = wer(rec.reference, baseline_text)

        if baseline_text.strip() == "":
            result = {
                "status": "baseline_empty_skip",
                "final_text": baseline_text,
                "used_adaptation": False,
                "loss": float("nan"),
                "grad_norm": float("nan"),
                "active_frames_before": 0,
                "active_frames_after": 0,
                "chars_before": 0,
                "chars_after": 0,
                "rollback": True,
            }
        else:
            t0 = time.perf_counter()
            result = adapt_one_clip(
                model=model,
                processor=processor,
                waveform=waveform,
                baseline_logits=baseline_logits,
                baseline_text=baseline_text,
                selected=selected,
                snapshot=snapshot,
                device=device,
                lr=args.lr,
                steps=args.steps,
                temperature=args.temperature,
                length_guard=args.length_guard,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            result["adapt_ms"] = (time.perf_counter() - t0) * 1000.0

        adapted_text = result.get("final_text", baseline_text)
        adapted_text_mapped = (
            colonial_to_modern_nahuatl(adapted_text)
            if adapted_text != baseline_text
            else adapted_text
        )
        adapted_wer = wer(rec.reference, adapted_text_mapped)

        row = {
            "index": i - 1,
            "dialect": rec.dialect,
            "dataset": rec.dataset,
            "audio_filename": rec.audio_filename,
            "reference": rec.reference,
            "baseline_text": baseline_text,
            "adapted_text": adapted_text_mapped,
            "baseline_wer": baseline_wer,
            "adapted_wer": adapted_wer,
            "delta_wer": adapted_wer - baseline_wer,
            "status": result.get("status", ""),
            "failure_reason": result.get("failure_reason", ""),
            "used_adaptation": bool(result.get("used_adaptation", False)),
            "rollback": bool(result.get("rollback", False)),
            "loss": result.get("loss", float("nan")),
            "grad_norm": result.get("grad_norm", float("nan")),
            "active_frames_before": result.get("active_frames_before", 0),
            "active_frames_after": result.get("active_frames_after", 0),
            "chars_before": result.get("chars_before", 0),
            "chars_after": result.get("chars_after", 0),
            "adapt_ms": result.get("adapt_ms", float("nan")),
        }
        rows.append(row)

        print(
            f"[{i:02d}/{len(records)}] "
            f"dialect={rec.dialect:<5} "
            f"baseWER={baseline_wer:.4f} "
            f"adaptWER={adapted_wer:.4f} "
            f"delta={adapted_wer - baseline_wer:+.4f} "
            f"chars={row['chars_before']}->{row['chars_after']} "
            f"status={row['status']}"
        )

        restore_trainable(selected, snapshot)
        if device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "suta_results.csv", index=False)

    summary = {
        "n_clips": int(len(df)),
        "baseline_mean_wer": float(df["baseline_wer"].mean()),
        "adapted_mean_wer": float(df["adapted_wer"].mean()),
        "mean_delta_wer": float(df["delta_wer"].mean()),
        "adapted_better_fraction": float((df["delta_wer"] < 0).mean()),
        "rollback_fraction": float(df["rollback"].mean()),
        "mean_grad_norm": float(df["grad_norm"].replace([np.inf, -np.inf], np.nan).mean()),
        "mean_adapt_ms": float(df["adapt_ms"].replace([np.inf, -np.inf], np.nan).mean()),
    }

    dialect_summary = {}
    for dialect, g in df.groupby("dialect"):
        dialect_summary[str(dialect)] = {
            "n": int(len(g)),
            "baseline_wer": float(g["baseline_wer"].mean()),
            "adapted_wer": float(g["adapted_wer"].mean()),
            "delta_wer": float(g["delta_wer"].mean()),
        }
    summary["by_dialect"] = dialect_summary

    with open(out_dir / "suta_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================ SUTA SUMMARY ================")
    print(f"Clips             : {len(df)}")
    print(f"Baseline WER      : {summary['baseline_mean_wer']:.6f}")
    print(f"Adapted WER       : {summary['adapted_mean_wer']:.6f}")
    print(f"Mean delta        : {summary['mean_delta_wer']:+.6f}")
    print(f"Clips improved    : {100.0 * summary['adapted_better_fraction']:.1f}%")
    print(f"Rollbacks/skips   : {100.0 * summary['rollback_fraction']:.1f}%")
    print(f"Mean adapt time   : {summary['mean_adapt_ms']:.1f} ms")

    print("\nBy dialect:")
    for dialect, vals in dialect_summary.items():
        print(
            f"  {dialect:<5} n={vals['n']:2d} "
            f"base={vals['baseline_wer']:.4f} "
            f"adapt={vals['adapted_wer']:.4f} "
            f"delta={vals['delta_wer']:+.4f}"
        )

    restore_trainable(selected, snapshot)
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-tsv", required=True)
    ap.add_argument("--corpus-audio-dir", required=True)
    ap.add_argument("--dev-tsv", required=True)
    ap.add_argument("--dev-audio-dir", required=True)

    ap.add_argument("--corpus-audio-col", default=None)
    ap.add_argument("--corpus-ref-col", default=None)
    ap.add_argument("--dev-audio-col", default=None)
    ap.add_argument("--dev-ref-col", default=None)
    ap.add_argument("--dev-lang-col", default=None)

    ap.add_argument("--safety-test", action="store_true")
    ap.add_argument("--full", action="store_true")

    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--length-guard", type=float, default=0.75)
    ap.add_argument("--adapt-head", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--out-dir", default="suta_output")

    args = ap.parse_args()

    if args.safety_test and args.full:
        raise ValueError("Use --safety-test OR --full, not both.")
    if not args.safety_test and not args.full:
        raise ValueError("Choose --safety-test or --full.")

    records = load_validation_records(args)
    print(
        f"[DATA] Loaded {len(records)} validation records "
        f"(nhi={sum(r.dialect == 'nhi' for r in records)}, "
        f"azz={sum(r.dialect == 'azz' for r in records)}, "
        f"nhw={sum(r.dialect == 'nhw' for r in records)})"
    )

    if args.safety_test:
        records = select_safety_subset(records)
        print("[DATA] Using stratified 10-clip safety set.")
    else:
        if len(records) != 635:
            raise RuntimeError(f"Expected exactly 635 records, got {len(records)}.")
        print("[DATA] Using all 635 held-out clips.")

    evaluate(records, args)


if __name__ == "__main__":
    main()