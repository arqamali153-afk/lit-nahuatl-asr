"""
audio_io.py

Loads a single audio file (WAV at train time, MP3 at inference time) and
returns 16kHz mono float32 samples, matching what MMS/Wav2Vec2-family
models expect. Deliberately a pure function of one file path -- no shared
buffers, no caching across clips, no global audio state.
"""

import torch
import torchaudio

TARGET_SR = 16000

# Safe, content-free diagnostic counters -- which backend actually served
# each clip. Not proof of correct decoding (a bad libsndfile MP3 decode
# wouldn't necessarily throw), but at least makes backend usage visible
# rather than fully silent, in case WER analysis later points here.
_backend_use_counts = {"soundfile": 0, "torchaudio_fallback": 0}


def load_audio_16k_mono(path: str):
    """
    Returns a 1D float32 numpy array at 16kHz, mono.

    Tries soundfile first to avoid TorchCodec / FFmpeg backend issues on
    local Windows environments, falling back to torchaudio.load() in
    environments where torchaudio has a native system backend configured.
    """
    waveform = None
    sr = None
    used_backend = None
    soundfile_error = None

    # 1. Primary loader: soundfile (handles WAV and MP3 without TorchCodec)
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32")
        tensor = torch.from_numpy(data)
        if tensor.ndim == 1:
            waveform = tensor.unsqueeze(0)
        else:
            # (samples, channels) -> (channels, samples)
            waveform = tensor.t()
        used_backend = "soundfile"
    except Exception as e:
        waveform = None
        soundfile_error = f"{type(e).__name__}: {e}"

    # 2. Fallback loader: torchaudio
    if waveform is None:
        try:
            waveform, sr = torchaudio.load(path)
            used_backend = "torchaudio_fallback"
        except Exception as e:
            # Both backends failed -- surface BOTH reasons, not just the
            # fallback's, since the primary's failure is usually the more
            # informative one and was previously being silently discarded.
            raise RuntimeError(
                f"Both audio backends failed for this clip. "
                f"soundfile error: {soundfile_error!r}. "
                f"torchaudio error: {type(e).__name__}: {e!r}."
            ) from None

    _backend_use_counts[used_backend] += 1

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    return waveform.squeeze(0).numpy()


def get_backend_use_summary() -> str:
    """Safe, aggregate-only summary (counts, not content) of which audio
    backend actually served clips this run. Call once at the end of a run
    (e.g. from main.py) to log it -- helps catch a scenario where the
    primary backend is silently mis-decoding a format without throwing."""
    return f"soundfile={_backend_use_counts['soundfile']}, torchaudio_fallback={_backend_use_counts['torchaudio_fallback']}"


def check_available_backends():
    """Safe diagnostic: checks available audio loading backends."""
    backends = []
    try:
        import soundfile as sf

        backends.append(f"soundfile (libsndfile {sf.__libsndfile_version__})")
    except Exception:
        pass

    try:
        if hasattr(torchaudio, "list_audio_backends"):
            backends.extend(torchaudio.list_audio_backends())
        elif hasattr(torchaudio, "backend") and hasattr(
            torchaudio.backend, "list_audio_backends"
        ):
            backends.extend(torchaudio.backend.list_audio_backends())
        else:
            backends.append("torchaudio (default)")
    except Exception as e:
        backends.append(f"<torchaudio error: {e}>")

    return ", ".join(backends) if backends else "none detected"