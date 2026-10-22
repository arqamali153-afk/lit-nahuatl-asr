"""
finetune_nhi_adapter.py

Self-contained Colab script (paste cell-by-cell, split on the `# %%`
markers, or run as one script) to fine-tune MMS-1B-all's `nhi` adapter on
your training corpus, using `original_sentence` as the label (validated
as already matching the competition's target orthography).

Strategy:
  - Build a tokenizer/vocab directly from YOUR training labels (guarantees
    every character you actually need is representable -- don't assume
    MMS's stock nhi vocab matches).
  - Initialize from the REAL pretrained nhi adapter (target_lang="nhi")
    as a head start, not from scratch.
  - Freeze the 1B backbone; train only the adapter attention layers +
    output head (small, fast, fits a T4).
  - fp16 + gradient checkpointing + grad accumulation to fit 15GB VRAM.

I have not been able to execute or verify this end-to-end (no GPU/HF
access in my environment) -- treat the "Sanity check" cell at the end as
mandatory, not optional, before trusting the saved adapter.
"""

# %% [Cell 1] Install dependencies
# !pip install -q transformers torch torchaudio jiwer accelerate safetensors scikit-learn

# %% [Cell 2] Config -- EDIT THESE PATHS for your Colab environment
CORPUS_TSV_PATH = "/content/drive/MyDrive/lost_in_transcription/data/tetelancingo_nahuatl/corpus.tsv"
AUDIO_DIR = "/content/drive/MyDrive/lost_in_transcription/data/tetelancingo_nahuatl/audios"
OUTPUT_DIR = "/content/nhi_adapter_finetuned"
BASE_MODEL_ID = "facebook/mms-1b-all"  # Colab has internet, so pull from Hub directly
TARGET_LANG = "nhi"

NUM_EPOCHS = 30          # small dataset -- more epochs than usual, watch eval WER for overfitting
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 2   # effective batch size 8
LEARNING_RATE = 3e-4
VAL_SPEAKER_FRACTION = 0.2    # held-out speakers for validation, NOT random-row split

# %% [Cell 3] Imports
import os
import json
import numpy as np
import pandas as pd
import torch
import torchaudio
import jiwer
from dataclasses import dataclass
from typing import Dict, List, Union
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)

# %% [Cell 4] Load corpus, keep only the provided 'train' split, group-split by speaker
df = pd.read_csv(CORPUS_TSV_PATH, sep="\t")
df = df[df["split"] == "train"].reset_index(drop=True)
df["original_sentence"] = df["original_sentence"].fillna("").astype(str)

gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SPEAKER_FRACTION, random_state=42)
train_idx, val_idx = next(gss.split(df, groups=df["speaker"]))
train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)

print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)} "
      f"| Train speakers: {train_df['speaker'].nunique()} "
      f"| Val speakers: {val_df['speaker'].nunique()}")
# Sanity check: these speaker sets must not overlap.
assert set(train_df["speaker"]).isdisjoint(set(val_df["speaker"])), "Speaker leakage between train/val!"

# %% [Cell 5] Build a vocab/tokenizer from YOUR training labels
def build_vocab(transcripts: List[str]) -> Dict[str, int]:
    chars = set()
    for t in transcripts:
        chars.update(t)
    chars.discard(" ")
    vocab_list = sorted(chars)
    vocab_dict = {c: i for i, c in enumerate(vocab_list)}
    vocab_dict["|"] = len(vocab_dict)       # word delimiter (standard CTC convention)
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    return vocab_dict

vocab_dict = build_vocab(train_df["original_sentence"].tolist())
print(f"Vocab size from training labels: {len(vocab_dict)}")

vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, ensure_ascii=False)

tokenizer = Wav2Vec2CTCTokenizer(
    vocab_path,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000, padding_value=0.0,
    do_normalize=True, return_attention_mask=True,
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# %% [Cell 6] Dataset
class NahuatlDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, audio_dir, processor):
        self.df = dataframe
        self.audio_dir = audio_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row["audio_file"])
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.squeeze(0).numpy()

        input_values = self.processor(waveform, sampling_rate=16000).input_values[0]
        with self.processor.as_target_processor():
            labels = self.processor(row["original_sentence"]).input_ids

        return {"input_values": input_values, "labels": labels}

train_dataset = NahuatlDataset(train_df, AUDIO_DIR, processor)
val_dataset = NahuatlDataset(val_df, AUDIO_DIR, processor)

# %% [Cell 7] Data collator (standard CTC padding pattern)
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=True, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor)

# %% [Cell 8] Load model -- initialize from real pretrained nhi adapter, then
# resize head to OUR vocab (guarantees full character coverage).
model = Wav2Vec2ForCTC.from_pretrained(
    BASE_MODEL_ID,
    target_lang=TARGET_LANG,          # loads real nhi backbone+adapter as starting point
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True,     # head resized to OUR vocab -- head is fresh, rest is pretrained
    pad_token_id=processor.tokenizer.pad_token_id,
    ctc_loss_reduction="mean",
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    layerdrop=0.0,
)
# LESSON LEARNED (see asr_model.py history): target_lang on from_pretrained
# only resizes the head -- it does NOT reliably guarantee the real adapter
# attention weights are loaded. Force it explicitly:
model.load_adapter(TARGET_LANG, force_load=True)
model.tie_weights()

# Freeze feature encoder (standard practice -- low-level CNN features
# transfer fine, don't need to be retrained on 3.4 hours of audio).
model.freeze_feature_encoder()

# Freeze everything except adapter layers + the output head. There is no
# single built-in "freeze_base_model()" method for this -- do it explicitly
# so it's correct regardless of transformers version:
trainable_params = 0
total_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    if ("adapter" in name) or ("lm_head" in name):
        param.requires_grad = True
        trainable_params += param.numel()
    else:
        param.requires_grad = False

print(f"Trainable params: {trainable_params:,} / {total_params:,} "
      f"({100 * trainable_params / total_params:.2f}%)")
assert trainable_params > 0, "No trainable parameters matched -- check param name patterns above."

model.gradient_checkpointing_enable()
model = model.to("cuda")

# %% [Cell 9] Metric
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    labels = pred.label_ids
    labels[labels == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(labels, group_tokens=False)

    wer = jiwer.wer(label_str, pred_str)
    return {"wer": wer}

# %% [Cell 10] Train
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    gradient_checkpointing=True,
    fp16=True,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=2,
    report_to=[],  # disable wandb/etc prompts
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

# %% [Cell 11] Save the fine-tuned adapter -- matches the filename convention
# asr_model.py already expects (adapter.nhi.safetensors), so it drops in
# with NO code changes needed there.
FINAL_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "final_adapter")
os.makedirs(FINAL_ADAPTER_DIR, exist_ok=True)
model.save_adapter(FINAL_ADAPTER_DIR, TARGET_LANG)
# Also save the vocab/tokenizer -- REQUIRED, since we trained a custom
# vocab that differs from the stock nhi vocab.json. This has to travel
# with the adapter or decoding will be wrong.
processor.save_pretrained(FINAL_ADAPTER_DIR)

print(f"Saved to {FINAL_ADAPTER_DIR}. Files:")
for f in os.listdir(FINAL_ADAPTER_DIR):
    print(" ", f)

# %% [Cell 12] MANDATORY sanity check -- reload fresh and verify non-degenerate
# output on a couple of your OWN held-out val examples before trusting this.
# Only print the WER NUMBER here if you want to share it back for review --
# do not paste the decoded text itself into any LLM chat.
del model
torch.cuda.empty_cache()

check_model = Wav2Vec2ForCTC.from_pretrained(
    BASE_MODEL_ID, target_lang=TARGET_LANG,
    vocab_size=len(processor.tokenizer), ignore_mismatched_sizes=True,
).to("cuda")
check_model.load_adapter(TARGET_LANG, force_load=True)
# Overwrite with our fine-tuned weights:
from safetensors.torch import load_file
finetuned_state = load_file(os.path.join(FINAL_ADAPTER_DIR, "adapter.nhi.safetensors"))
missing, unexpected = check_model.load_state_dict(finetuned_state, strict=False)
print(f"Sanity reload -- missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

check_model.eval()
sample = val_dataset[0]
with torch.no_grad():
    logits = check_model(torch.tensor(sample["input_values"]).unsqueeze(0).to("cuda")).logits
pred_ids = torch.argmax(logits, dim=-1)
decoded = processor.batch_decode(pred_ids)[0]
reference = val_df.iloc[0]["original_sentence"]
single_wer = jiwer.wer(reference, decoded)
print(f"Sanity check single-sample WER: {single_wer:.3f}  (0.0=perfect match, 1.0=no words match)")
print("If this is near 1.0 or decoded text looks empty/garbage, do NOT ship this adapter yet.")
