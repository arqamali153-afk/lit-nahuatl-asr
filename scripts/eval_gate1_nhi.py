"""
eval_gate1_nhi.py

Gate 1 of the guardrail plan: benchmark the NEW fine-tuned nhi adapter
(now sitting at weights/mms-1b-all/adapter.nhi.safetensors, replacing
stock) against real nhi dev clips and real ground truth, using the
ACTUAL competition scorer (score.py) for a trustworthy number.

Tests both raw and colonial-mapped output -- the adapter was trained on
the `sentence` column (colonial orthography, same convention as stock),
so the mapping is still expected to help on top of it, same as before.

Compares against your previously-recorded stock baseline (measured via
verify_mapping_with_real_scorer.py, same 25-clip sample/seed) since the
stock adapter file has been overwritten in place and can't be re-measured
live unless you kept a backup copy.

PASS/FAIL: per your own guardrail plan, this must clearly beat the stock
raw WER (0.6427) to proceed to Gate 2 (free Smoke Test).

Run from the project root:
    python eval_gate1_nhi.py
"""

import os
import sys

sys.path.append("src")
sys.path.append(".")

import pandas as pd
import jiwer

from audio_io import load_audio_16k_mono
from asr_model import transcribe_stock_forced, _ensure_stock_adapter, _run_inference, warm_up
from score import normalize_text  # the REAL scorer's normalizer

METADATA_PATH = "data/nahuatl_dev/metadata.tsv"
METADATA_SEP = "\t"
CLIPS_DIR = "data/nahuatl_dev/clips"
AUDIO_FILENAME_COLUMN = "audio_filename"
LABEL_COLUMN = "transcript"
LANGUAGE_COLUMN = "language"

SAMPLE_SIZE = 25
RANDOM_SEED = 42  # same seed as verify_mapping_with_real_scorer.py -- same sample, fair comparison

# Historical reference, measured previously on the STOCK (pre-finetune)
# nhi adapter with this same script/seed. Not re-measurable live since
# the file has been overwritten -- restore a backup copy if you have one
# for a true simultaneous comparison instead of this hardcoded reference.
HISTORICAL_STOCK_RAW_WER = 0.6427
HISTORICAL_STOCK_MAPPED_WER = 0.6129


def real_wer(refs, preds):
    norm_refs = [normalize_text(r) for r in refs]
    norm_preds = [normalize_text(p) for p in preds]
    return jiwer.wer(norm_refs, norm_preds)


def main():
    df = pd.read_csv(METADATA_PATH, sep=METADATA_SEP)
    nhi_df = df[df[LANGUAGE_COLUMN] == "nhi"].reset_index(drop=True)
    if len(nhi_df) == 0:
        print("No rows with language=='nhi' found -- check column names/values.")
        return

    n = min(SAMPLE_SIZE, len(nhi_df))
    sample = nhi_df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Evaluating on {n} real nhi dev clips (of {len(nhi_df)} available).")

    print("Loading model (now includes the fine-tuned nhi adapter, drop-in replaced)...")
    warm_up()

    refs, raw_preds, mapped_preds = [], [], []
    for i, row in sample.iterrows():
        audio_path = os.path.join(CLIPS_DIR, row[AUDIO_FILENAME_COLUMN])
        waveform = load_audio_16k_mono(audio_path)

        model, processor = _ensure_stock_adapter("nhi")
        raw_text = _run_inference(model, processor, waveform, apply_colonial_mapping=False)
        mapped_text = _run_inference(model, processor, waveform, apply_colonial_mapping=True)

        refs.append(str(row[LABEL_COLUMN]))
        raw_preds.append(raw_text)
        mapped_preds.append(mapped_text)

        if (i + 1) % 5 == 0 or (i + 1) == n:
            print(f"  Processed {i + 1}/{n}")

    wer_raw = real_wer(refs, raw_preds)
    wer_mapped = real_wer(refs, mapped_preds)

    print("\n=== GATE 1 RESULTS (real scorer, n={}) ===".format(n))
    print(f"  Fine-tuned nhi, raw (no mapping):     {wer_raw:.4f}")
    print(f"  Fine-tuned nhi, colonial-mapped:       {wer_mapped:.4f}")
    print(f"  --- vs historical stock (same sample/seed) ---")
    print(f"  Stock nhi, raw (no mapping):           {HISTORICAL_STOCK_RAW_WER:.4f}")
    print(f"  Stock nhi, colonial-mapped:             {HISTORICAL_STOCK_MAPPED_WER:.4f}")
    print(f"\n  Delta (raw):    {HISTORICAL_STOCK_RAW_WER - wer_raw:+.4f} (positive = fine-tuned wins)")
    print(f"  Delta (mapped): {HISTORICAL_STOCK_MAPPED_WER - wer_mapped:+.4f} (positive = fine-tuned wins)")

    best_finetuned = min(wer_raw, wer_mapped)
    best_stock = min(HISTORICAL_STOCK_RAW_WER, HISTORICAL_STOCK_MAPPED_WER)
    print(f"\n  GATE 1: {'PASS' if best_finetuned < best_stock else 'FAIL'} "
          f"(best fine-tuned {best_finetuned:.4f} vs best stock {best_stock:.4f})")
    print("  If PASS: proceed to Gate 2 (free Smoke Test on the espnah-dominated pool).")
    print("  If FAIL: do not spend a submission on this adapter.")


if __name__ == "__main__":
    main()