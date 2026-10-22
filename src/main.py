"""
main.py - Step 1: packaging-first, plumbing-only pipeline for the
Lost In Transcription (Spanish-Nahuatl) code-execution submission.

Goal of this version: prove the container runs end-to-end inside
DrivenData's runtime -- read the manifest, "process" every clip with a
placeholder, and write a correctly formatted submission.csv -- BEFORE any
real modeling happens. No model is loaded here on purpose, so this can be
Smoke Tested immediately and cheaply.

Hard constraints encoded here (per competition rules):
  - No network access is available or used.
  - Strict test-sample independence: transcribe_clip() is a pure function
    of a single clip (+ its own manifest row); it never sees other rows
    or files, and nothing here accumulates cross-sample state.
  - No retraining / warm-up adaptation at inference time.
  - Never print transcript text, clip durations, or test-set statistics
    to stdout -- only coarse milestone counts.
  - Output CSV is written via pandas.to_csv so commas/quotes in transcripts
    are handled with correct RFC-4180 quoting automatically.
  - A single bad/missing clip must not crash the whole run.
"""

import os
import sys
import traceback

import pandas as pd

from audio_io import load_audio_16k_mono, check_available_backends, get_backend_use_summary
from asr_model import transcribe as model_transcribe, warm_up as model_warm_up
from orthography import normalize_nahuatl_orthography

# ---------------------------------------------------------------------------
# Execution topology. Overridable via env vars for local dry runs; defaults
# match the documented runtime paths exactly.
# ---------------------------------------------------------------------------
WORKING_DIR = os.environ.get("CODE_EXECUTION_WORKING_DIR", ".")
MANIFEST_PATH = os.path.join(WORKING_DIR, "data", "test_metadata.csv")
CLIPS_DIR = os.path.join(WORKING_DIR, "data", "clips")
OUTPUT_DIR = os.path.join(WORKING_DIR, "submission")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "submission.csv")


def log(msg: str) -> None:
    """Coarse milestone logging only.

    NEVER pass transcript text, clip durations, filenames-with-content, or
    any other per-sample data into this function -- only counts and fixed
    status strings. Logs are capped at 500 lines / 300 chars upstream.
    """
    print(f"[INFO] {msg}", flush=True)


def load_manifest(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found at expected path: {path}")
    df = pd.read_csv(path)
    if "audio_filename" not in df.columns:
        raise ValueError("Manifest is missing required 'audio_filename' column.")
    return df


def transcribe_clip(clip_path: str, language: str = None) -> str:
    """
    Real transcription step: decode+resample the clip, run it through the
    dialect-routed MMS model, apply orthography post-processing.

    Contract preserved:
      - Pure function of (clip_path, language) only.
      - No access to any other clip, row, or the manifest as a whole.
      - Model weights are loaded once globally (see warm_up() in run()) --
        that's shared *infrastructure*, not shared *data*; nothing about
        this prediction depends on any other sample's audio or output.
    """
    waveform = load_audio_16k_mono(clip_path)
    raw_text = model_transcribe(waveform, language=language)
    return normalize_nahuatl_orthography(raw_text)


def run() -> None:
    log("Starting submission pipeline.")

    import torch
    import torchaudio
    import transformers
    log(f"Torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    log(f"Torchaudio: {torchaudio.__version__}")
    log(f"Transformers: {transformers.__version__}")
    log(f"Available torchaudio backends: {check_available_backends()}")

    # Feasibility check for a possible future beam-search/LM decoding pass
    # -- safe to log (library availability, not data). Do this before
    # investing engineering time in kenlm/pyctcdecode integration.
    try:
        import pyctcdecode
        log(f"pyctcdecode available: YES (loaded from {pyctcdecode.__file__})")
    except Exception as e:
        log(f"pyctcdecode NOT available: {e}")

    try:
        import kenlm
        log("kenlm available: YES (compiled C++ extension present)")
    except Exception as e:
        log(f"kenlm NOT available: {e}")

    df = load_manifest(MANIFEST_PATH)
    log(f"Loaded manifest with {len(df)} rows.")

    # Load model weights once, up front. If this fails, fail loudly and
    # exit now rather than retrying (and re-logging) inside every
    # per-clip try/except below -- a broken model makes the whole run
    # invalid regardless, so a missing output file is the correct signal.
    log("Loading model weights.")
    model_warm_up()
    log("Model weights loaded.")

    has_language_col = "language" in df.columns
    predictions = []
    first_error_logged = False

    total = len(df)
    log_every = max(1, total // 10)  # roughly 10 progress lines, any size

    for i, row in enumerate(df.itertuples(index=False)):
        audio_filename = getattr(row, "audio_filename")
        language = getattr(row, "language", None) if has_language_col else None
        clip_path = os.path.join(CLIPS_DIR, audio_filename)

        try:
            transcript = transcribe_clip(clip_path, language=language)
        except Exception as e:
            # A single bad clip must never crash the whole run -- fall back
            # to an empty (but valid) prediction. But silently swallowing
            # every error makes failures undiagnosable, so log the first
            # one's type+message (safe: library/codec error text, never
            # transcript content) and stay quiet on repeats to respect the
            # log line cap.
            transcript = ""
            if not first_error_logged:
                log(f"WARNING: a clip failed to transcribe ({type(e).__name__}: {e}). "
                    f"Further per-clip failures this run will not be individually logged.")
                first_error_logged = True

        predictions.append({"audio_filename": audio_filename, "transcript": transcript})

        if (i + 1) % log_every == 0 or (i + 1) == total:
            log(f"Processed {i + 1}/{total} clips.")

    out_df = pd.DataFrame(predictions, columns=["audio_filename", "transcript"])

    # Safe, content-free diagnostic: a count of empty predictions tells us
    # a lot (e.g. "0/5" vs "5/5" points at very different problems) without
    # ever printing any transcript text or clip-level detail.
    empty_count = int((out_df["transcript"].fillna("").str.strip() == "").sum())
    log(f"Empty/blank predictions: {empty_count}/{len(out_df)}.")
    log(f"Audio backend usage: {get_backend_use_summary()}.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # pandas handles RFC-4180 quoting for fields containing commas/quotes --
    # never hand-build this CSV with string interpolation.
    out_df.to_csv(OUTPUT_PATH, index=False)

    log(f"Wrote submission file with {len(out_df)} rows to {OUTPUT_PATH}.")


if __name__ == "__main__":
    try:
        run()
    except Exception:
        log("FATAL: pipeline failed. See bounded traceback below.")
        traceback.print_exc(limit=10)  # trace only, never sample content
        sys.exit(1)