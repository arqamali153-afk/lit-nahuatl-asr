"""
verify_mapping_with_real_scorer.py (Memory-Safe CPU version)
"""

import os
import sys
import gc

sys.path.append("src")
sys.path.append(".")

import torch
import pandas as pd
import jiwer

from audio_io import load_audio_16k_mono
from asr_model import _ensure_stock_adapter, _run_inference, warm_up
from orthography import colonial_to_modern_nahuatl
from score import normalize_text

METADATA_PATH = "data/nahuatl_dev/metadata.tsv"
METADATA_SEP = "\t"
CLIPS_DIR = "data/nahuatl_dev/clips"
AUDIO_FILENAME_COLUMN = "audio_filename"
LABEL_COLUMN = "transcript"
LANGUAGE_COLUMN = "language"

# Keep at 15-20 per dialect to stay well within RAM limits on CPU
SAMPLE_SIZE_PER_DIALECT = 15
RANDOM_SEED = 42
DIALECTS = ["nhi", "azz", "nhw"]


def real_wer(refs, preds):
    norm_refs = [normalize_text(r) for r in refs]
    norm_preds = [normalize_text(p) for p in preds]
    return jiwer.wer(norm_refs, norm_preds)


def main():
    # Force single-thread or bounded threads to prevent CPU thread memory explosion
    torch.set_num_threads(4)
    df = pd.read_csv(METADATA_PATH, sep=METADATA_SEP)
    warm_up()

    for dialect in DIALECTS:
        subset = df[df[LANGUAGE_COLUMN] == dialect]
        if len(subset) == 0:
            print(f"[{dialect}] No rows found.")
            continue

        n = min(SAMPLE_SIZE_PER_DIALECT, len(subset))
        sample = subset.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)

        refs, raw_preds, mapped_preds = [], [], []

        print(f"\nProcessing {dialect} ({n} clips)...")
        for idx, row in sample.iterrows():
            audio_path = os.path.join(CLIPS_DIR, row[AUDIO_FILENAME_COLUMN])
            waveform = load_audio_16k_mono(audio_path)

            model, processor = _ensure_stock_adapter(dialect)

            # Strict memory guard: inference_mode disables all autograd overhead
            with torch.inference_mode():
                raw_text = _run_inference(model, processor, waveform, apply_colonial_mapping=False)

            mapped_text = colonial_to_modern_nahuatl(raw_text)

            refs.append(str(row[LABEL_COLUMN]))
            raw_preds.append(raw_text)
            mapped_preds.append(mapped_text)

            del waveform
            if idx % 5 == 0:
                gc.collect()

        wer_raw = real_wer(refs, raw_preds)
        wer_mapped = real_wer(refs, mapped_preds)

        print(f"[{dialect}] Results (Real Platform Scorer):")
        print(f"  Raw (no mapping):     {wer_raw:.4f}")
        print(f"  Colonial-mapped:      {wer_mapped:.4f}")
        print(f"  Delta (raw - mapped): {wer_raw - wer_mapped:+.4f}")


if __name__ == "__main__":
    main()