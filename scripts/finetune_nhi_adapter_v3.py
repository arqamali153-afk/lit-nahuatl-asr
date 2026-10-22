"""
finetune_nhi_adapter_v3.py

Replicates the methodology from Pugh et al. 2025 ("Ihquin tlahtouah in
Tetelahtzincocah") on YOUR copy of the same corpus: stock vocab (no
custom tokenizer/lm_head rebuild), many more epochs than the previous
attempt, early-stopped on eval WER.

RECOMMENDED PLATFORM: Kaggle, not Colab -- "Save & Run All" (Commit) runs
headless in the cloud for up to 9-12 hours with no browser tab needed, no
risk of losing the run to a dropped connection or a sleeping laptop
(both of which cost real time earlier in this project). Kaggle also
mounts input datasets on fast local NVMe (no Google Drive FUSE stall).

Setup on Kaggle:
  1. Zip your local tetelancingo_nahuatl/ folder (corpus.tsv + audios/),
     upload as a new PRIVATE Kaggle dataset named e.g. "tetelancingo-nahuatl".
  2. New Notebook -> right panel: Accelerator = GPU P100 (or T4 x1),
     Internet = On (needed once, to pull base MMS weights from the Hub),
     Add Input -> attach your dataset.
  3. Check the exact mounted path in the Input file tree (usually
     /kaggle/input/<dataset-name>/...) and set CORPUS_TSV_PATH/AUDIO_DIR
     below to match.
  4. Save Version -> Save & Run All (Commit). Close the tab if you want --
     check back later under the notebook's "Versions" tab. Final
     adapter.nhi.safetensors lands in /kaggle/working/, downloadable once
     the run completes.

Guardrail plan (unchanged): Gate 1 = local benchmark on held-out nhi dev
clips (must clearly beat stock 0.6136). Gate 2 = free Smoke Test on the
platform (must beat 0.5385). Only spend a real submission if both pass.
Per the paper's own held-out-speaker numbers, expect something closer to
WER ~0.42 as a realistic floor, not the more optimistic same-corpus-split
0.38 -- and even that may not fully transfer to the real competition
test set's different speakers/conditions/register.
"""

# %% [Cell 1] Install dependencies
# !pip install -q transformers torch torchaudio jiwer accelerate safetensors scikit-learn

# %% [Cell 2] Config -- EDIT THESE PATHS for Kaggle (see setup notes above)
CORPUS_TSV_PATH = "/kaggle/input/tetelancingo-nahuatl/corpus.tsv"
AUDIO_DIR = "/kaggle/input/tetelancingo-nahuatl/audios"
OUTPUT_DIR = "/kaggle/working/nhi_adapter_v3"
BASE_MODEL_ID = "facebook/mms-1b-all"
TARGET_LANG = "nhi"

MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 6
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 2
LEARNING_RATE = 3e-4
VAL_SPEAKER_FRACTION = 0.2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %% [Cell 3] Imports
import unicodedata
import numpy as np
import pandas as pd
import torch
import torchaudio
import jiwer
from dataclasses import dataclass
from typing import Dict, List, Union
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoProcessor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# %% [Cell 4] Load corpus, speaker-grouped split
df = pd.read_csv(CORPUS_TSV_PATH, sep="\t")
df = df[df["split"] == "train"].reset_index(drop=True)
df["sentence"] = df["sentence"].fillna("").astype(str)

gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SPEAKER_FRACTION, random_state=42)
train_idx, val_idx = next(gss.split(df, groups=df["speaker"]))
train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)

print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)} "
      f"| Train speakers: {train_df['speaker'].nunique()} "
      f"| Val speakers: {val_df['speaker'].nunique()}")
assert set(train_df["speaker"]).isdisjoint(set(val_df["speaker"])), "Speaker leakage!"

# Speaker distribution check -- with only 5 speakers total, test_size=0.2
# reserves exactly ~1 speaker for validation. Verify that speaker isn't
# a tiny or overwhelming slice before trusting eval WER / early stopping.
print("\nTrain speaker distribution:")
print(train_df["speaker"].value_counts())
print("\nVal speaker distribution:")
print(val_df["speaker"].value_counts())

# %% [Cell 5] Load the STOCK processor/tokenizer -- no custom vocab rebuild
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, target_lang=TARGET_LANG)
stock_vocab = set(processor.tokenizer.get_vocab().keys())
print(f"\nStock {TARGET_LANG} vocab size: {len(stock_vocab)}")
print(f"Vocab characters: {sorted(stock_vocab)}")

import re

_SENTENCE_PUNCT = re.compile(r"[.,;:!?\"()\[\]]")


def clean_text_to_vocab(text: str, vocab: set) -> str:
    """Two-pass cleaning:
    1. Strip standard sentence-level punctuation to SPACE (these sit
       between words -- dropping them without a space would wrongly
       fuse separate words together).
    2. Strip accents, then for any character STILL not in the model's
       real vocab (mostly mid-word marks like apostrophes/saltillo at
       this point), drop it with EMPTY STRING, not space -- these occur
       INSIDE words, so replacing with a space would wrongly split a
       single word into two, inflating WER artificially.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _SENTENCE_PUNCT.sub(" ", text)

    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))

    text = "".join(c if (c in vocab or c == " ") else "" for c in text)
    text = " ".join(text.split())
    return text


train_df["clean_sentence"] = train_df["sentence"].apply(lambda s: clean_text_to_vocab(s, stock_vocab))
val_df["clean_sentence"] = val_df["sentence"].apply(lambda s: clean_text_to_vocab(s, stock_vocab))

# CRITICAL: filter out rows that end up empty after cleaning -- an empty
# label produces a zero-length CTC target, which reliably produces
# NaN/inf loss during backprop and silently wrecks the adapter weights.
before_train, before_val = len(train_df), len(val_df)
train_df = train_df[train_df["clean_sentence"].str.strip() != ""].reset_index(drop=True)
val_df = val_df[val_df["clean_sentence"].str.strip() != ""].reset_index(drop=True)
print(f"\nFiltered empty-after-cleaning rows -- train: {before_train} -> {len(train_df)}, "
      f"val: {before_val} -> {len(val_df)}")

# %% [Cell 5b] Pre-load and pre-resample ALL audio into memory once.
# Corpus is small (~3.4 hours total, well under 1GB as float32 mono
# 16kHz) -- caching in RAM eliminates repeat CPU resampling on every
# __getitem__ call across up to 40 epochs, which would otherwise leave
# the GPU waiting on I/O the whole run.
def load_and_resample(path: str) -> np.ndarray:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    return waveform.squeeze(0).numpy()


print("\nPre-loading and resampling all audio into memory...")
audio_cache: Dict[str, np.ndarray] = {}
for fname in pd.concat([train_df["audio_file"], val_df["audio_file"]]).unique():
    audio_cache[fname] = load_and_resample(os.path.join(AUDIO_DIR, fname))
print(f"Cached {len(audio_cache)} unique audio files in memory.")

# %% [Cell 6] Dataset -- reads from the in-memory cache, no disk I/O per step
class NahuatlDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, audio_cache, processor):
        self.df = dataframe
        self.audio_cache = audio_cache
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform = self.audio_cache[row["audio_file"]]

        input_values = self.processor(waveform, sampling_rate=16000).input_values[0]
        labels = self.processor.tokenizer(row["clean_sentence"]).input_ids

        return {"input_values": input_values, "labels": labels}


train_dataset = NahuatlDataset(train_df, audio_cache, processor)
val_dataset = NahuatlDataset(val_df, audio_cache, processor)

# %% [Cell 7] Data collator -- mask by pad_token_id directly rather than
# relying on an attention_mask that Wav2Vec2/MMS CTC tokenizers don't
# consistently produce on tokenizer.pad() output.
@dataclass
class DataCollatorCTCWithPadding:
    processor: object

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
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

# %% [Cell 8] Model -- stock vocab, stock head, no resize
model = Wav2Vec2ForCTC.from_pretrained(
    BASE_MODEL_ID,
    target_lang=TARGET_LANG,
    ignore_mismatched_sizes=True,   # no-op here, shapes already match stock
    pad_token_id=processor.tokenizer.pad_token_id,
    ctc_loss_reduction="mean",
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    layerdrop=0.0,
)
model.load_adapter(TARGET_LANG, force_load=True)
model.tie_weights()

model.freeze_feature_encoder()

trainable_params = 0
total_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    if ("adapter" in name) or ("lm_head" in name):
        param.requires_grad = True
        trainable_params += param.numel()
    else:
        param.requires_grad = False

print(f"\nTrainable params: {trainable_params:,} / {total_params:,} "
      f"({100 * trainable_params / total_params:.2f}%)")
assert trainable_params > 0

# CRITICAL: with a frozen backbone + gradient checkpointing, PyTorch needs
# an explicit signal that gradients should flow from the checkpointed
# segments back to the (frozen-input) trainable adapter layers, or it
# raises "element 0 of tensors does not require grad and does not have a
# grad_fn". This call is that signal.
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model = model.to("cuda")

# %% [Cell 9] Metric -- defensively guards against any empty ground truth
# slipping through (shouldn't happen after Cell 5's filter, but jiwer
# raises hard on empty references, so don't let one bad batch kill eval).
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    labels = pred.label_ids
    labels[labels == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(labels, group_tokens=False)

    valid_pairs = [(ref, hyp) for ref, hyp in zip(label_str, pred_str) if ref.strip() != ""]
    if not valid_pairs:
        return {"wer": 1.0}
    clean_refs, clean_hyps = zip(*valid_pairs)
    wer = jiwer.wer(list(clean_refs), list(clean_hyps))
    return {"wer": wer}

# %% [Cell 10] Train
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    gradient_checkpointing=True,
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=processor.feature_extractor,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

trainer.train()

# %% [Cell 11] Save -- same shape as stock, drop-in replacement for
# weights/mms-1b-all/adapter.nhi.safetensors, no asr_model.py changes needed
FINAL_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "final_adapter")
os.makedirs(FINAL_ADAPTER_DIR, exist_ok=True)

from safetensors.torch import save_file
adapter_state = {
    name: param.detach().cpu()
    for name, param in model.state_dict().items()
    if ("adapter" in name) or ("lm_head" in name)
}
save_file(adapter_state, os.path.join(FINAL_ADAPTER_DIR, "adapter.nhi.safetensors"))

print(f"\nSaved to {FINAL_ADAPTER_DIR}/adapter.nhi.safetensors")
print("Drop-in replacement for weights/mms-1b-all/adapter.nhi.safetensors -- no other code changes needed.")

# %% [Cell 12] MANDATORY sanity check before downloading
del model
torch.cuda.empty_cache()

check_model = Wav2Vec2ForCTC.from_pretrained(
    BASE_MODEL_ID, target_lang=TARGET_LANG, ignore_mismatched_sizes=True,
).to("cuda")
# Explicit call before load_state_dict -- without this, the adapter
# submodules may not be allocated in the model's parameter structure at
# all, causing load_state_dict to silently dump fine-tuned weights into
# unexpected_keys instead of actually applying them.
check_model.load_adapter(TARGET_LANG, force_load=True)

from safetensors.torch import load_file
finetuned_state = load_file(os.path.join(FINAL_ADAPTER_DIR, "adapter.nhi.safetensors"))
missing, unexpected = check_model.load_state_dict(finetuned_state, strict=False)
print(f"Sanity reload -- missing keys: {len(missing)} (expected: backbone), "
      f"unexpected keys: {len(unexpected)} (expected: 0)")

check_model.eval()
sample = val_dataset[0]
with torch.no_grad():
    logits = check_model(torch.tensor(sample["input_values"]).unsqueeze(0).to("cuda")).logits
pred_ids = torch.argmax(logits, dim=-1)
decoded = processor.batch_decode(pred_ids)[0]
reference = val_df.iloc[0]["clean_sentence"]
single_wer = jiwer.wer(reference, decoded)
print(f"Sanity check single-sample WER: {single_wer:.3f}")
print("If near 1.0 or output looks empty/garbage, do NOT ship this adapter.")