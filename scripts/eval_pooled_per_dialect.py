"""
eval_pooled_per_dialect.py

Runs the per-dialect breakdown that finetune_pooled_adapter.py's own
Cell would have run, but standalone -- against the recovered checkpoint
from the timed-out run, using the REAL scorer.

This is the missing piece: the pooled run's own printed 0.4510 is a
COMBINED number, likely dominated by nhi's larger validation split. This
tells us what actually happened to azz/nhw specifically -- the entire
point of pooling.

Loading is defensive: works whether the recovered file is adapter-only
or a full model state dict (load_state_dict(strict=False) tolerates
extra keys either way, so no special-casing needed for which one you have).

Does NOT touch weights/mms-1b-all/adapter.nhi.safetensors -- loads the
candidate checkpoint into a separate model instance for evaluation only,
so your current production adapter (the validated nhi-only one) stays
untouched until this candidate is confirmed better.

Run from the project root:
    python eval_pooled_per_dialect.py /path/to/recovered/adapter.nhi.safetensors
"""

import os
import sys

sys.path.append("src")
sys.path.append(".")

import re
import unicodedata

import jiwer
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoProcessor, Wav2Vec2ForCTC

from audio_io import load_audio_16k_mono
from score import normalize_text

BASE_MODEL_ID = "facebook/mms-1b-all"
MMS_WEIGHTS_DIR = "src/weights/mms-1b-all"  # local config/tokenizer files, same as production
HOST_LANG = "nhi"

CORPUS_TSV_PATH = "data/tetelancingo_nahuatl/corpus.tsv"
TETELANCINGO_AUDIO_DIR = "data/tetelancingo_nahuatl/audios"
DEV_METADATA_PATH = "data/nahuatl_dev/metadata.tsv"
DEV_CLIPS_DIR = "data/nahuatl_dev/clips"

VAL_SPEAKER_FRACTION = 0.2

if len(sys.argv) < 2:
    print("Usage: python eval_pooled_per_dialect.py /path/to/recovered/adapter.nhi.safetensors")
    sys.exit(1)
CANDIDATE_ADAPTER_PATH = sys.argv[1]

_SENTENCE_PUNCT = re.compile(r"[.,;:!?\"()\[\]]")


def clean_text_to_vocab(text: str, vocab: set) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _SENTENCE_PUNCT.sub(" ", text)
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    text = "".join(c if (c in vocab or c == " ") else "" for c in text)
    return " ".join(text.split())


def real_wer(refs, preds):
    return jiwer.wer([normalize_text(r) for r in refs], [normalize_text(p) for p in preds])


# %% Rebuild the SAME held-out validation splits used during training
print("Rebuilding held-out validation splits (same logic/seeds as training)...")

tet_df = pd.read_csv(CORPUS_TSV_PATH, sep="\t")
tet_df = tet_df[tet_df["split"] == "train"].reset_index(drop=True)
tet_df["sentence"] = tet_df["sentence"].fillna("").astype(str)
gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SPEAKER_FRACTION, random_state=42)
_, val_idx = next(gss.split(tet_df, groups=tet_df["speaker"]))
nhi_val = tet_df.iloc[val_idx].reset_index(drop=True)
nhi_val["audio_path"] = nhi_val["audio_file"].apply(lambda f: os.path.join(TETELANCINGO_AUDIO_DIR, f))
nhi_val["raw_text"] = nhi_val["sentence"]

dev_df = pd.read_csv(DEV_METADATA_PATH, sep="\t")
label_col = "transcript" if "transcript" in dev_df.columns else "sentence"
audio_col = "audio_filename" if "audio_filename" in dev_df.columns else "audio_file"
dev_df[label_col] = dev_df[label_col].fillna("").astype(str)
dev_df["audio_path"] = dev_df[audio_col].apply(lambda f: os.path.join(DEV_CLIPS_DIR, f))
dev_df["raw_text"] = dev_df[label_col]
has_speaker_col = "speaker" in dev_df.columns


def split_dialect_val_only(dialect_name):
    d = dev_df[dev_df["language"] == dialect_name].reset_index(drop=True)
    unique_speakers = d["speaker"].nunique() if has_speaker_col else 0
    if has_speaker_col and unique_speakers > 1:
        gss_d = GroupShuffleSplit(n_splits=1, test_size=VAL_SPEAKER_FRACTION, random_state=42)
        _, va_idx = next(gss_d.split(d, groups=d["speaker"]))
        return d.iloc[va_idx].reset_index(drop=True)
    else:
        shuffled = d.sample(frac=1.0, random_state=42).reset_index(drop=True)
        val_n = max(1, int(len(shuffled) * VAL_SPEAKER_FRACTION))
        return shuffled.iloc[:val_n].reset_index(drop=True)


azz_val = split_dialect_val_only("azz")
nhw_val = split_dialect_val_only("nhw")

val_by_dialect = {"nhi": nhi_val, "azz": azz_val, "nhw": nhw_val}

# %% Load host processor + candidate model
print(f"\nLoading host ({HOST_LANG}) processor and candidate checkpoint...")
processor = AutoProcessor.from_pretrained(MMS_WEIGHTS_DIR, target_lang=HOST_LANG, local_files_only=True)
stock_vocab = set(processor.tokenizer.get_vocab().keys())

model = Wav2Vec2ForCTC.from_pretrained(
    MMS_WEIGHTS_DIR, target_lang=HOST_LANG, ignore_mismatched_sizes=True, local_files_only=True,
)
model.load_adapter(HOST_LANG, force_load=True)

candidate_state = load_file(CANDIDATE_ADAPTER_PATH)
missing, unexpected = model.load_state_dict(candidate_state, strict=False)
print(f"Loaded candidate -- missing keys: {len(missing)}, unexpected keys: {len(unexpected)} "
      f"(unexpected>0 is fine if this came from a full Trainer checkpoint -- extra backbone "
      f"keys are just ignored)")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# %% Per-dialect evaluation
print("\n=== PER-DIALECT VALIDATION WER (real scorer) ===")
for dialect, val_df in val_by_dialect.items():
    if len(val_df) == 0:
        print(f"  {dialect}: no validation data")
        continue
    refs, hyps = [], []
    for _, row in val_df.iterrows():
        waveform = load_audio_16k_mono(row["audio_path"])
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        hyp = processor.batch_decode(pred_ids)[0]
        ref = clean_text_to_vocab(row["raw_text"], stock_vocab)
        refs.append(ref)
        hyps.append(hyp)
    wer = real_wer(refs, hyps)
    print(f"  {dialect}: n={len(val_df)}  WER={wer:.4f}")

print("\nCompare each dialect's number above against its stock/nhi-only-tuned baseline "
      "before deciding whether this pooled checkpoint is worth Gate 2 (Smoke Test).")
