"""
test_blank_bias_local.py

Tests a targeted bias on ONLY the CTC blank token's logit, before argmax
-- NOT temperature scaling (which is mathematically inert under greedy
argmax, since it's a monotonic transform of ALL logits equally and can't
change which index wins). Biasing just the blank token changes its
ranking relative to real characters at each frame, which CAN change
greedy output: a negative bias makes blank less likely (may reduce
under-transcription/missed characters), a positive bias makes it more
likely (may reduce insertions/hallucinated repeats).

Sweeps a range of bias values across ALL THREE dialects' full held-out
validation sets (537/74/24 -- the same splits from eval_pooled_per_dialect.py),
not the 5-clip smoke sample, to avoid overfitting to a tiny pool per
Goodhart's Law.

In HF's Wav2Vec2ForCTC, config.pad_token_id IS the CTC blank symbol by
convention -- confirmed via processor.tokenizer.pad_token_id.

Run from the project root:
    python test_blank_bias_local.py
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
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoProcessor, Wav2Vec2ForCTC

from audio_io import load_audio_16k_mono
from score import normalize_text

MMS_WEIGHTS_DIR = "src/weights/mms-1b-all"  # production weights -- already has the pooled+soup adapter
HOST_LANG = "nhi"

CORPUS_TSV_PATH = "data/tetelancingo_nahuatl/corpus.tsv"
TETELANCINGO_AUDIO_DIR = "data/tetelancingo_nahuatl/audios"
DEV_METADATA_PATH = "data/nahuatl_dev/metadata.tsv"
DEV_CLIPS_DIR = "data/nahuatl_dev/clips"
VAL_SPEAKER_FRACTION = 0.2

BIAS_VALUES_TO_TEST = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]  # 0.0 = current (no bias) baseline

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


# %% Rebuild the same held-out validation splits used throughout
print("Rebuilding held-out validation splits...")
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
    shuffled = d.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_n = max(1, int(len(shuffled) * VAL_SPEAKER_FRACTION))
    return shuffled.iloc[:val_n].reset_index(drop=True)


azz_val = split_dialect_val_only("azz")
nhw_val = split_dialect_val_only("nhw")
val_by_dialect = {"nhi": nhi_val, "azz": azz_val, "nhw": nhw_val}

# %% Load model once
print(f"\nLoading production model (host={HOST_LANG})...")
processor = AutoProcessor.from_pretrained(MMS_WEIGHTS_DIR, target_lang=HOST_LANG, local_files_only=True)
stock_vocab = set(processor.tokenizer.get_vocab().keys())
model = Wav2Vec2ForCTC.from_pretrained(
    MMS_WEIGHTS_DIR, target_lang=HOST_LANG, ignore_mismatched_sizes=True, local_files_only=True,
)
model.load_adapter(HOST_LANG, force_load=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

blank_id = processor.tokenizer.pad_token_id
print(f"Blank/pad token id: {blank_id}")

# %% Pre-compute logits ONCE per clip (expensive part), reuse across all
# bias values in the sweep (cheap: just re-argmax with different bias).
print("\nPre-computing logits for all validation clips (this is the slow part, done once)...")
cached_logits = {}  # (dialect, idx) -> (logits tensor, ref text)
for dialect, val_df in val_by_dialect.items():
    for i, row in val_df.iterrows():
        waveform = load_audio_16k_mono(row["audio_path"])
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits[0].cpu()  # (T, vocab_size)
        ref = clean_text_to_vocab(row["raw_text"], stock_vocab)
        cached_logits[(dialect, i)] = (logits, ref)
    print(f"  {dialect}: {len(val_df)} clips cached")

# %% Sweep bias values, reusing cached logits (fast)
print("\n=== BLANK-TOKEN BIAS SWEEP (real scorer, full held-out sets) ===")
results = {}
for bias in BIAS_VALUES_TO_TEST:
    per_dialect_wer = {}
    for dialect in val_by_dialect:
        refs, hyps = [], []
        for (d, i), (logits, ref) in cached_logits.items():
            if d != dialect:
                continue
            biased = logits.clone()
            biased[:, blank_id] -= bias  # NOTE: subtracting `bias` makes blank
                                          # MORE likely to be picked as bias
                                          # goes negative, LESS likely as it
                                          # goes positive (since a bigger
                                          # subtraction lowers the blank
                                          # logit -- double check sign
                                          # matches your intent when reading
                                          # results below)
            pred_ids = torch.argmax(biased, dim=-1).unsqueeze(0)
            hyp = processor.batch_decode(pred_ids)[0]
            refs.append(ref)
            hyps.append(hyp)
        per_dialect_wer[dialect] = real_wer(refs, hyps)
    results[bias] = per_dialect_wer
    print(f"  bias={bias:+.1f}  nhi={per_dialect_wer['nhi']:.4f}  "
          f"azz={per_dialect_wer['azz']:.4f}  nhw={per_dialect_wer['nhw']:.4f}")

print("\nbias=0.0 is your current (no-bias) baseline -- compare every other row against it.")
print("Only adopt a bias value if it helps (or is neutral on) ALL THREE dialects --")
print("a bias that helps one dialect while hurting another is not a safe production change,")
print("since the real test set's exact mix under 'espnah' is unknown.")
