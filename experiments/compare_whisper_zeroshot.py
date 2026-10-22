"""
compare_whisper_zeroshot.py

Zero-shot comparison: OpenAI Whisper vs your current MMS stock baseline,
on the SAME real dev clips, scored with the REAL competition normalizer
(score.py) for a fair, directly comparable number.

Whisper has no Nahuatl exposure at all (unlike MMS, which was explicitly
pretrained on low-resource/indigenous languages including Nahuatl
varieties) -- but it IS trained on a much larger, more diverse mix of
real-world conversational/noisy audio, including plenty of Spanish. This
tests whether that generalist strength outweighs MMS's targeted-language
pretraining on THIS specific task (conversational Spanish-Nahuatl
code-switching).

Forces language="es" (Spanish) since Whisper has no Nahuatl language
code -- code-switched Nahuatl words will likely come through as Spanish-
phonetic-nearest guesses, which is a fair test of whether Whisper's
overall robustness compensates.

Usage:
    pip install openai-whisper   # if not already available
    python compare_whisper_zeroshot.py
"""

import os
import sys

sys.path.append("src")
sys.path.append(".")

import pandas as pd
import jiwer
import whisper

from audio_io import load_audio_16k_mono
from score import normalize_text  # the REAL scorer's normalizer

METADATA_PATH = "data/nahuatl_dev/metadata.tsv"
METADATA_SEP = "\t"
CLIPS_DIR = "data/nahuatl_dev/clips"
AUDIO_FILENAME_COLUMN = "audio_filename"
LABEL_COLUMN = "transcript"
LANGUAGE_COLUMN = "language"

WHISPER_MODEL_SIZE = "small"   # bump to "medium" or "large-v3" if you have GPU time to spare
WHISPER_FORCE_LANGUAGE = "es"  # Whisper has no Nahuatl code -- Spanish is the closest fit

SAMPLE_SIZE_PER_DIALECT = 25
RANDOM_SEED = 42
DIALECTS = ["nhi", "azz", "nhw"]


def real_wer(refs, preds):
    norm_refs = [normalize_text(r) for r in refs]
    norm_preds = [normalize_text(p) for p in preds]
    return jiwer.wer(norm_refs, norm_preds)


def main():
    print(f"Loading Whisper ({WHISPER_MODEL_SIZE})...")
    model = whisper.load_model(WHISPER_MODEL_SIZE)

    df = pd.read_csv(METADATA_PATH, sep=METADATA_SEP)

    for dialect in DIALECTS:
        subset = df[df[LANGUAGE_COLUMN] == dialect]
        if len(subset) == 0:
            print(f"[{dialect}] No rows found -- check column names/values.")
            continue

        n = min(SAMPLE_SIZE_PER_DIALECT, len(subset))
        sample = subset.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)

        refs, preds = [], []
        for i, row in sample.iterrows():
            audio_path = os.path.join(CLIPS_DIR, row[AUDIO_FILENAME_COLUMN])
            waveform = load_audio_16k_mono(audio_path)  # already 16kHz mono float32

            result = model.transcribe(
                waveform, language=WHISPER_FORCE_LANGUAGE, fp16=False
            )
            preds.append(result["text"])
            refs.append(str(row[LABEL_COLUMN]))

            if (i + 1) % 10 == 0 or (i + 1) == n:
                print(f"  [{dialect}] {i + 1}/{n}")

        wer = real_wer(refs, preds)
        print(f"\n[{dialect}] n={n}  Whisper-{WHISPER_MODEL_SIZE} zero-shot WER (real scorer): {wer:.4f}")

    print("\nCompare each number above against your MMS raw zero-shot numbers from "
          "verify_mapping_with_real_scorer.py's 'Raw (no mapping)' rows.")


if __name__ == "__main__":
    main()
