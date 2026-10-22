"""
asr_model.py

CONFIGURATION:
  - Beam Search: ON (pyctcdecode + KenLM, gentle alpha/beta)
  - Colonial Mapping: ON everywhere (main path + espnah fallback)

NOT YET VALIDATED: this specific alpha/beta + Spanish-inclusive corpus
combination has not been benchmarked locally or smoke-tested yet. Do
that before considering a real submission -- two prior beam-search
attempts (full LM alpha=0.5, and alpha=beta=0) both regressed against
greedy (0.6264 and 0.6044 vs 0.5385).
"""

import gc
import os
import sys
import torch
import numpy as np
from transformers import (
    Wav2Vec2ForCTC,
    AutoProcessor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)
from safetensors.torch import load_file

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from orthography import colonial_to_modern_nahuatl

# Optional dependency detection
try:
    import pyctcdecode
    _PYCTCDECODE_AVAILABLE = True
except ImportError:
    _PYCTCDECODE_AVAILABLE = False

try:
    import kenlm
    _KENLM_AVAILABLE = True
except ImportError:
    _KENLM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths and Devices
# ---------------------------------------------------------------------------
MMS_WEIGHTS_DIR = os.environ.get(
    "MMS_WEIGHTS_DIR", os.path.join(_THIS_DIR, "weights", "mms-1b-all")
)
FINETUNED_NHI_DIR = os.environ.get(
    "FINETUNED_NHI_DIR",
    os.path.join(_THIS_DIR, "weights", "nhi_adapter_finetuned", "final_adapter"),
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ---------------------------------------------------------------------------
# Diagnostic Experiment Toggles
# ---------------------------------------------------------------------------
STOCK_LANGUAGE_TO_ADAPTER = {
    "nhi": "nhi",
    "azz": "azz",
    "nhw": "nhw",
}
STOCK_FALLBACK_ADAPTER = "nhi"

# --- CONFIG FLAGS ---
USE_BEAM_SEARCH = False
APPLY_COLONIAL_MAPPING = True
STOCK_FALLBACK_APPLY_COLONIAL_MAPPING = True

BEAM_WIDTH = 32
LM_ALPHA = 0.05    # gentle LM prior -- previous attempts used 0.5 (too aggressive) and 0.0 (no LM signal at all)
LM_BETA = 0.1      # minimal word-insertion bonus, to avoid re-triggering artificial splitting

_unmapped_languages_seen = set()
_mapped_languages_seen = set()

# Model state
_stock_model = None
_stock_processor = None
_stock_current_adapter_lang = None
_decoders = {}

# Fine-tuned state
_nhi_model = None
_nhi_processor = None


def _prepare_labels_for_pyctcdecode(tokenizer):
    vocab = tokenizer.get_vocab()
    sorted_tokens = [k for k, _ in sorted(vocab.items(), key=lambda x: x[1])]
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    word_delimiter = getattr(tokenizer, "word_delimiter_token", "|")

    labels = []
    seen = set()
    for idx, token in enumerate(sorted_tokens):
        if idx == pad_id:
            lbl = ""
        elif token == word_delimiter:
            lbl = " "
        elif len(token) > 1:
            lbl = chr(idx + 1)
        else:
            lbl = token

        if lbl in seen:
            lbl = chr(0xE000 + idx)
        labels.append(lbl)
        seen.add(lbl)

    return labels


def _build_beam_decoder(processor, adapter_code: str):
    if not _PYCTCDECODE_AVAILABLE:
        return None

    labels = _prepare_labels_for_pyctcdecode(processor.tokenizer)
    unigrams_path = os.path.join(MMS_WEIGHTS_DIR, "unigrams.txt")
    if not os.path.exists(unigrams_path):
        unigrams_path = os.path.join(_THIS_DIR, "..", "data", "unigrams.txt")

    unigrams = None
    if os.path.exists(unigrams_path):
        with open(unigrams_path, "r", encoding="utf-8") as f:
            unigrams = [line.strip() for line in f if line.strip()]

    lm_binary_path = os.path.join(MMS_WEIGHTS_DIR, "nahuatl.binary")
    if not os.path.exists(lm_binary_path):
        lm_binary_path = os.path.join(_THIS_DIR, "..", "data", "nahuatl.binary")

    kenlm_path = lm_binary_path if (_KENLM_AVAILABLE and os.path.exists(lm_binary_path)) else None

    try:
        return pyctcdecode.build_ctcdecoder(
            labels=labels,
            kenlm_model_path=kenlm_path,
            unigrams=unigrams,
            alpha=LM_ALPHA,
            beta=LM_BETA,
        )
    except Exception as e:
        print(f"[ERROR] Failed to build CTC decoder for '{adapter_code}': {e}")
        return None


def _get_decoder(adapter_code: str, processor):
    if not USE_BEAM_SEARCH or not _PYCTCDECODE_AVAILABLE:
        return None
    if adapter_code not in _decoders:
        _decoders[adapter_code] = _build_beam_decoder(processor, adapter_code)
    return _decoders[adapter_code]


def _load_stock_model_and_processor():
    global _stock_model, _stock_processor, _stock_current_adapter_lang, _nhi_model
    if DEVICE == "cpu" and _nhi_model is not None:
        _nhi_model = None
        gc.collect()

    if _stock_model is None:
        initial_lang = next(iter(STOCK_LANGUAGE_TO_ADAPTER.values()))

        _stock_model = Wav2Vec2ForCTC.from_pretrained(
            MMS_WEIGHTS_DIR,
            target_lang=initial_lang,
            ignore_mismatched_sizes=True,
            local_files_only=True,
            dtype=TORCH_DTYPE,
        ).to(DEVICE)
        _stock_model.eval()
        _stock_processor = AutoProcessor.from_pretrained(
            MMS_WEIGHTS_DIR,
            target_lang=initial_lang,
            local_files_only=True,
        )
        _stock_model.load_adapter(initial_lang)
        _stock_processor.tokenizer.set_target_lang(initial_lang)
        _stock_current_adapter_lang = initial_lang

        if USE_BEAM_SEARCH:
            _get_decoder(initial_lang, _stock_processor)

    return _stock_model, _stock_processor


def _ensure_stock_adapter(adapter_code: str):
    global _stock_current_adapter_lang
    model, processor = _load_stock_model_and_processor()

    if adapter_code != _stock_current_adapter_lang:
        model.load_adapter(adapter_code)
        processor.tokenizer.set_target_lang(adapter_code)
        _stock_current_adapter_lang = adapter_code

    return model, processor


def _load_finetuned_nhi():
    global _nhi_model, _nhi_processor, _stock_model
    if DEVICE == "cpu" and _stock_model is not None:
        _stock_model = None
        gc.collect()

    if _nhi_model is None:
        nhi_tokenizer = Wav2Vec2CTCTokenizer(
            os.path.join(FINETUNED_NHI_DIR, "vocab.json"),
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
        )
        nhi_feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0,
            do_normalize=True, return_attention_mask=True,
        )
        _nhi_processor = Wav2Vec2Processor(
            feature_extractor=nhi_feature_extractor, tokenizer=nhi_tokenizer
        )
        custom_vocab_size = len(nhi_tokenizer)

        _nhi_model = Wav2Vec2ForCTC.from_pretrained(
            MMS_WEIGHTS_DIR,
            target_lang="nhi",
            ignore_mismatched_sizes=True,
            local_files_only=True,
            dtype=TORCH_DTYPE,
        )

        _nhi_model.config.vocab_size = custom_vocab_size
        _nhi_model.lm_head = torch.nn.Linear(
            _nhi_model.config.hidden_size, custom_vocab_size
        ).to(dtype=_nhi_model.dtype)
        _nhi_model = _nhi_model.to(DEVICE)

        finetuned_weights_path = os.path.join(FINETUNED_NHI_DIR, "adapter.nhi.safetensors")
        finetuned_state = load_file(finetuned_weights_path)
        missing, unexpected = _nhi_model.load_state_dict(finetuned_state, strict=False)
        print(f"[INFO] Fine-tuned nhi adapter loaded -- missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")

        _nhi_model.eval()
    return _nhi_model, _nhi_processor


def _run_inference(model, processor, waveform_16k_mono, apply_colonial_mapping: bool) -> str:
    inputs = processor(waveform_16k_mono, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device=DEVICE, dtype=model.dtype)

    with torch.inference_mode():
        logits = model(input_values).logits

    decoder = None
    if USE_BEAM_SEARCH and processor is _stock_processor:
        decoder = _get_decoder(_stock_current_adapter_lang, processor)

    if decoder is not None:
        try:
            logits_np = logits[0].cpu().float().numpy()
            decoded_text = decoder.decode(logits_np, beam_width=BEAM_WIDTH)
        except Exception:
            predicted_ids = torch.argmax(logits, dim=-1)
            decoded_text = processor.batch_decode(predicted_ids)[0]
    else:
        predicted_ids = torch.argmax(logits, dim=-1)
        decoded_text = processor.batch_decode(predicted_ids)[0]

    if apply_colonial_mapping:
        return colonial_to_modern_nahuatl(decoded_text)
    return decoded_text


@torch.inference_mode()
def transcribe(waveform_16k_mono, language: str) -> str:
    if language in STOCK_LANGUAGE_TO_ADAPTER:
        if language not in _mapped_languages_seen:
            print(f"[INFO] First clip seen with language tag {language!r}")
            _mapped_languages_seen.add(language)
        model, processor = _ensure_stock_adapter(STOCK_LANGUAGE_TO_ADAPTER[language])
        return _run_inference(model, processor, waveform_16k_mono,
                              apply_colonial_mapping=APPLY_COLONIAL_MAPPING)
    else:
        if language not in _unmapped_languages_seen:
            print(f"[WARNING] Unmapped language tag {language!r} -- routing to stock "
                  f"'{STOCK_FALLBACK_ADAPTER}' adapter, colonial_mapping="
                  f"{STOCK_FALLBACK_APPLY_COLONIAL_MAPPING}.")
            _unmapped_languages_seen.add(language)
        model, processor = _ensure_stock_adapter(STOCK_FALLBACK_ADAPTER)
        return _run_inference(model, processor, waveform_16k_mono,
                              apply_colonial_mapping=STOCK_FALLBACK_APPLY_COLONIAL_MAPPING)


@torch.inference_mode()
def transcribe_stock_forced(waveform_16k_mono, adapter_code: str) -> str:
    model, processor = _ensure_stock_adapter(adapter_code)
    return _run_inference(model, processor, waveform_16k_mono,
                          apply_colonial_mapping=APPLY_COLONIAL_MAPPING)


@torch.inference_mode()
def transcribe_finetuned_forced(waveform_16k_mono) -> str:
    model, processor = _load_finetuned_nhi()
    return _run_inference(model, processor, waveform_16k_mono, apply_colonial_mapping=False)


def warm_up():
    stock_model, stock_processor = _load_stock_model_and_processor()
    stock_vocab_size = len(stock_processor.tokenizer)
    stock_head_size = stock_model.lm_head.out_features
    print(
        f"[INFO] Stock model warm-up -- active_adapter={_stock_current_adapter_lang}, "
        f"tokenizer_vocab_size={stock_vocab_size}, lm_head_out_features={stock_head_size}"
    )
    print(
        f"[DIAGNOSTIC CONFIG] beam_search_active={USE_BEAM_SEARCH}, "
        f"colonial_mapping_active={APPLY_COLONIAL_MAPPING or STOCK_FALLBACK_APPLY_COLONIAL_MAPPING}, "
        f"lm_alpha={LM_ALPHA}, lm_beta={LM_BETA}, beam_width={BEAM_WIDTH}"
    )