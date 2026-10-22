"""
finetune_pooled_adapter.py

Trains ONE adapter exposed to all three dialects, to serve as the single
espnah fallback -- the actual real-world target, given the test manifest
never disambiguates azz/nhw/nhi.

DESIGN DECISIONS:
  - Host vocab: nhi's stock 40-token vocab. Building a new union vocab
    across dialects would repeat the exact failure mode from the FIRST
    fine-tune attempt (custom vocab + reinitialized head + too little
    training). azz/nhw text is mapped onto nhi's vocab via the same
    accent-strip-then-drop cleaning already validated in the nhi run.
  - Oversampling, not hard balance: azz/nhw are moderately oversampled
    (duplicated) toward TARGET_DIALECT_FRACTIONS below, not forced to an
    exact 1:1:1 split -- with only ~130/~100 real clips each, heavy
    duplication doesn't add new information, it just means more epochs
    on the same tiny set (which a single-dialect run could do anyway).
    The actual goal here is avoiding catastrophic collapse on non-nhi
    phonetics, not achieving azz/nhw-specific excellence.
  - Per-dialect validation, reported separately: a pooled aggregate WER
    could hide "great on nhi, still bad on azz" -- this script evaluates
    each dialect's held-out slice independently after training.

Run on Kaggle: attach BOTH the tetelancingo-nahuatl dataset AND a
nahuatl-dev dataset (metadata.tsv + clips/) as inputs.
"""

import os

BASE_MODEL_ID = "facebook/mms-1b-all"
HOST_LANG = "nhi"  # the vocab/head all three dialects get mapped onto

MAX_EPOCHS = 16
EARLY_STOPPING_PATIENCE = 6
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUMULATION_STEPS = 4
LEARNING_RATE = 3e-4
VAL_SPEAKER_FRACTION = 0.2
MAX_CLIP_DURATION_SECONDS = 20.0

# Target per-epoch representation. nhi naturally dominates (2,681 vs
# ~130/~100 clips) -- these fractions control how much azz/nhw get
# oversampled toward, via row duplication. Adjust freely; this is the
# single easiest knob to revisit if results suggest a different balance.
TARGET_DIALECT_FRACTIONS = {"nhi": 0.5, "azz": 0.25, "nhw": 0.25}

OUTPUT_DIR = "/kaggle/working/pooled_adapter"
FINAL_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "final_adapter")

# --- Auto-detect both dataset paths in /kaggle/input ---
CORPUS_TSV_PATH = None
TETELANCINGO_AUDIO_DIR = None
DEV_METADATA_PATH = None
DEV_CLIPS_DIR = None

for root, dirs, files in os.walk("/kaggle/input"):
    if "corpus.tsv" in files and CORPUS_TSV_PATH is None:
        CORPUS_TSV_PATH = os.path.join(root, "corpus.tsv")
    if "audios" in dirs and TETELANCINGO_AUDIO_DIR is None:
        TETELANCINGO_AUDIO_DIR = os.path.join(root, "audios")
    if "metadata.tsv" in files and DEV_METADATA_PATH is None:
        DEV_METADATA_PATH = os.path.join(root, "metadata.tsv")
    if "clips" in dirs and DEV_CLIPS_DIR is None:
        DEV_CLIPS_DIR = os.path.join(root, "clips")

missing = [name for name, val in [
    ("corpus.tsv", CORPUS_TSV_PATH), ("audios/", TETELANCINGO_AUDIO_DIR),
    ("dev metadata.tsv", DEV_METADATA_PATH), ("dev clips/", DEV_CLIPS_DIR),
] if val is None]
if missing:
    tree = [f"{r} -> dirs: {d} | files: {f[:3]}" for r, d, f in os.walk("/kaggle/input")]
    raise FileNotFoundError(f"Missing: {missing}\n\nStructure:\n" + "\n".join(tree))

print(f"Tetelancingo corpus: {CORPUS_TSV_PATH}")
print(f"Tetelancingo audio:  {TETELANCINGO_AUDIO_DIR}")
print(f"Dev metadata:        {DEV_METADATA_PATH}")
print(f"Dev clips:           {DEV_CLIPS_DIR}")

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Union

import jiwer
import numpy as np
import pandas as pd
import torch
import torchaudio
from safetensors.torch import load_file, save_file
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Wav2Vec2ForCTC,
)

# %% Load nhi (Tetelancingo) -- speaker-grouped split, as before
tet_df = pd.read_csv(CORPUS_TSV_PATH, sep="\t")
tet_df = tet_df[tet_df["split"] == "train"].reset_index(drop=True)
tet_df["sentence"] = tet_df["sentence"].fillna("").astype(str)
tet_df["dialect"] = "nhi"
tet_df["audio_path"] = tet_df["audio_file"].apply(lambda f: os.path.join(TETELANCINGO_AUDIO_DIR, f))
tet_df["raw_text"] = tet_df["sentence"]

gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SPEAKER_FRACTION, random_state=42)
train_idx, val_idx = next(gss.split(tet_df, groups=tet_df["speaker"]))
nhi_train = tet_df.iloc[train_idx].reset_index(drop=True)
nhi_val = tet_df.iloc[val_idx].reset_index(drop=True)
print(f"nhi: train={len(nhi_train)}, val={len(nhi_val)} "
      f"(val speakers: {nhi_val['speaker'].unique().tolist()})")

# %% Load azz/nhw from nahuatl_dev
dev_df = pd.read_csv(DEV_METADATA_PATH, sep="\t")
label_col = "transcript" if "transcript" in dev_df.columns else "sentence"
audio_col = "audio_filename" if "audio_filename" in dev_df.columns else "audio_file"
duration_col = next((c for c in ["file_duration_seconds", "duration", "duration_seconds"]
                     if c in dev_df.columns), None)

dev_df[label_col] = dev_df[label_col].fillna("").astype(str)
dev_df["audio_path"] = dev_df[audio_col].apply(lambda f: os.path.join(DEV_CLIPS_DIR, f))
dev_df["raw_text"] = dev_df[label_col]

if duration_col:
    before = len(dev_df)
    dev_df = dev_df[dev_df[duration_col] <= MAX_CLIP_DURATION_SECONDS].reset_index(drop=True)
    print(f"Duration filter: {before} -> {len(dev_df)}")

has_speaker_col = "speaker" in dev_df.columns


def split_dialect(dialect_name):
    d = dev_df[dev_df["language"] == dialect_name].reset_index(drop=True)
    d["dialect"] = dialect_name

    unique_speakers = d["speaker"].nunique() if has_speaker_col else 0

    # Only group-split if there are at least 2 distinct speakers to divide
    # between train/val -- with exactly 1 (or 0 detected), GroupShuffleSplit
    # crashes (ValueError: resulting train set will be empty), since holding
    # out the only speaker for validation leaves none for training.
    if has_speaker_col and unique_speakers > 1:
        gss_d = GroupShuffleSplit(n_splits=1, test_size=VAL_SPEAKER_FRACTION, random_state=42)
        tr_idx, va_idx = next(gss_d.split(d, groups=d["speaker"]))
        tr, va = d.iloc[tr_idx].reset_index(drop=True), d.iloc[va_idx].reset_index(drop=True)
    else:
        print(f"NOTE: {dialect_name} has {unique_speakers} unique speaker(s) -- "
              f"falling back to a random row split. IMPORTANT: with only one "
              f"speaker available, no split can give genuine speaker-held-out "
              f"validation for this dialect -- train and val will share the "
              f"same voice regardless. Treat this dialect's resulting WER as "
              f"weaker evidence than nhi's (which does have real speaker "
              f"separation), not directly comparable evidence of the same strength.")
        shuffled = d.sample(frac=1.0, random_state=42).reset_index(drop=True)
        val_n = max(1, int(len(shuffled) * VAL_SPEAKER_FRACTION))
        va, tr = shuffled.iloc[:val_n].reset_index(drop=True), shuffled.iloc[val_n:].reset_index(drop=True)

    print(f"{dialect_name}: train={len(tr)}, val={len(va)}")
    return tr, va


azz_train, azz_val = split_dialect("azz")
nhw_train, nhw_val = split_dialect("nhw")

if len(azz_train) < 30 or len(nhw_train) < 30:
    print("WARNING: very little azz/nhw training data available -- expect high variance.")

# %% Host vocab (nhi) -- map ALL dialects' text onto it
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, target_lang=HOST_LANG)
stock_vocab = set(processor.tokenizer.get_vocab().keys())
print(f"\nHost ({HOST_LANG}) vocab size: {len(stock_vocab)}")

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


def prep(df):
    df = df.copy()
    df["clean_sentence"] = df["raw_text"].apply(lambda s: clean_text_to_vocab(s, stock_vocab))
    return df[df["clean_sentence"].str.strip() != ""].reset_index(drop=True)


nhi_train, nhi_val = prep(nhi_train), prep(nhi_val)
azz_train, azz_val = prep(azz_train), prep(azz_val)
nhw_train, nhw_val = prep(nhw_train), prep(nhw_val)

for name, before, after in [
    ("nhi", len(tet_df[tet_df["dialect"] == "nhi"]), len(nhi_train) + len(nhi_val)),
]:
    pass  # (informational only -- main filter counts already printed per split above)

# %% Oversample azz/nhw toward TARGET_DIALECT_FRACTIONS via row duplication
train_parts = {"nhi": nhi_train, "azz": azz_train, "nhw": nhw_train}
total_natural = sum(len(df) for df in train_parts.values())
print(f"\nNatural train totals: " + ", ".join(f"{k}={len(v)}" for k, v in train_parts.items()))

# Target count for each dialect, anchored to whichever dialect's natural
# count best satisfies its own target fraction without needing to shrink
# anything (we only ever duplicate up, never drop real data).
anchor_total = max(len(df) / frac for df, frac in
                   ((train_parts[k], TARGET_DIALECT_FRACTIONS[k]) for k in train_parts) if frac > 0)

oversampled_parts = []
for name, df in train_parts.items():
    target_count = int(round(anchor_total * TARGET_DIALECT_FRACTIONS[name]))
    if len(df) == 0:
        continue
    reps = target_count // len(df)
    remainder = target_count % len(df)
    pieces = [df] * max(reps, 1)
    if remainder > 0:
        pieces.append(df.sample(n=remainder, random_state=42))
    oversampled = pd.concat(pieces, ignore_index=True)
    print(f"  {name}: {len(df)} natural -> {len(oversampled)} after oversampling "
          f"(~{len(oversampled) / len(df):.1f}x)")
    oversampled_parts.append(oversampled)

train_df = pd.concat(oversampled_parts, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"\nPooled training set: {len(train_df)} rows "
      f"({train_df['dialect'].value_counts().to_dict()})")

# Validation stays UN-oversampled and SEPARATE per dialect -- we want to
# know true per-dialect performance, not a duplication-weighted blend.
val_by_dialect = {"nhi": nhi_val, "azz": azz_val, "nhw": nhw_val}

# %% Pre-cache all audio (train + every val split) into RAM
def load_and_resample(path: str) -> np.ndarray:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    return waveform.squeeze(0).numpy()


print("\nPre-loading and resampling audio into RAM (unique files only, "
      "so oversampled duplicates don't re-decode)...")
all_paths = pd.concat(
    [train_df["audio_path"]] + [v["audio_path"] for v in val_by_dialect.values()]
).unique()
audio_cache: Dict[str, np.ndarray] = {p: load_and_resample(p) for p in all_paths}
print(f"Cached {len(audio_cache)} unique audio files.")


class PooledDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, audio_cache, processor):
        self.df = dataframe
        self.audio_cache = audio_cache
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform = self.audio_cache[row["audio_path"]]
        input_values = self.processor(waveform, sampling_rate=16000).input_values[0]
        labels = self.processor.tokenizer(row["clean_sentence"]).input_ids
        return {"input_values": input_values, "labels": labels}


train_dataset = PooledDataset(train_df, audio_cache, processor)
# Single combined val set for Trainer's own eval_strategy="epoch" tracking
# (used for early stopping only) -- per-dialect breakdown happens
# separately after training, below.
combined_val_df = pd.concat(list(val_by_dialect.values()), ignore_index=True)
val_dataset = PooledDataset(combined_val_df, audio_cache, processor)


@dataclass
class DataCollatorCTCWithPadding:
    processor: object

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, padding=True, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )
        batch["labels"] = labels
        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor)

# %% Model
model = Wav2Vec2ForCTC.from_pretrained(
    BASE_MODEL_ID, target_lang=HOST_LANG, ignore_mismatched_sizes=True,
    pad_token_id=processor.tokenizer.pad_token_id, ctc_loss_reduction="mean",
)
model.load_adapter(HOST_LANG, force_load=True)
model.tie_weights()
model.freeze_feature_encoder()

trainable_params, total_params = 0, 0
for name, param in model.named_parameters():
    total_params += param.numel()
    if ("adapter" in name) or ("lm_head" in name):
        param.requires_grad = True
        trainable_params += param.numel()
    else:
        param.requires_grad = False
print(f"\nTrainable: {trainable_params:,} / {total_params:,} "
      f"({100 * trainable_params / total_params:.2f}%)")
assert trainable_params > 0

model = model.to("cuda")


def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    labels = pred.label_ids
    labels[labels == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(labels, group_tokens=False)
    valid = [(r, h) for r, h in zip(label_str, pred_str) if r.strip() != ""]
    if not valid:
        return {"wer": 1.0}
    refs, hyps = zip(*valid)
    return {"wer": jiwer.wer(list(refs), list(hyps))}


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    gradient_checkpointing=False,
    fp16=True,
    num_train_epochs=MAX_EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=2,
    report_to=[],
)

trainer_kwargs = dict(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
    data_collator=data_collator, compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)
try:
    trainer = Trainer(**trainer_kwargs, processing_class=processor.feature_extractor)
except TypeError:
    trainer = Trainer(**trainer_kwargs, tokenizer=processor.feature_extractor)

print("Starting pooled training on Kaggle GPU...")
trainer.train()

# %% Save -- shape matches nhi's stock adapter; drop-in replacement for
# weights/mms-1b-all/adapter.nhi.safetensors, used as the espnah fallback
os.makedirs(FINAL_ADAPTER_DIR, exist_ok=True)
adapter_state = {
    name: param.detach().cpu() for name, param in model.state_dict().items()
    if ("adapter" in name) or ("lm_head" in name)
}
adapter_out_path = os.path.join(FINAL_ADAPTER_DIR, "adapter.nhi.safetensors")
save_file(adapter_state, adapter_out_path)
print(f"\nSaved pooled adapter to: {adapter_out_path}")

# %% MANDATORY: per-dialect breakdown, not just pooled aggregate
model.eval()
print("\n=== PER-DIALECT VALIDATION WER (held-out, no oversampling) ===")
for dialect, val_df in val_by_dialect.items():
    if len(val_df) == 0:
        print(f"  {dialect}: no validation data available")
        continue
    refs, hyps = [], []
    for _, row in val_df.iterrows():
        waveform = audio_cache[row["audio_path"]]
        input_values = processor(waveform, sampling_rate=16000, return_tensors="pt").input_values.to("cuda")
        with torch.no_grad():
            logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        hyps.append(processor.batch_decode(pred_ids)[0])
        refs.append(row["clean_sentence"])
    wer = jiwer.wer(refs, hyps)
    print(f"  {dialect}: n={len(val_df)}  WER={wer:.4f}")

print("\nCompare each dialect's WER above against its own stock baseline "
      "before deciding whether this pooled adapter beats the current "
      "single-dialect nhi-tuned one on non-nhi content specifically.")
