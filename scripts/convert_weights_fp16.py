"""
convert_weights_fp16.py

Run this ONCE, locally, after download_mms_weights.py. Converts the
downloaded fp32 safetensors files (model backbone + adapters) to fp16 in
place, roughly halving their size. The A100 in the runtime container
supports fp16 natively, and MMS/Wav2Vec2-family models tolerate fp16
inference well.

Memory note: converts each file's tensors IN PLACE (mutating the same
dict) rather than building a second parallel dict -- this roughly halves
peak RAM usage during conversion, since the old fp32 tensor for a given
key is freed as soon as its fp16 replacement is written, rather than
holding two full-size dicts simultaneously. This matters most for
model.safetensors (~4GB fp32) -- if you ever re-download it fresh (fp32,
not the shrunk fp16 version), the old "build a second dict" approach
could momentarily need 6GB+ RAM, which can be silently OOM-killed by
Windows with no Python traceback at all (looks like "fails with no
error").

Usage:
    python convert_weights_fp16.py
"""

import gc
import os
import sys
import traceback
import torch
from safetensors.torch import load_file, save_file

WEIGHTS_DIR = os.path.join("src", "weights", "mms-1b-all")

FILES_TO_CONVERT = [
    "model.safetensors",
    "adapter.nhi.safetensors",
    "adapter.azz.safetensors",
    "adapter.nhw.safetensors",
    "adapter.spa.safetensors",
]


def convert_file(path: str) -> None:
    before_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Loading {os.path.basename(path)} ({before_mb:.1f} MB)...", flush=True)

    tensors = load_file(path)

    print(f"  Converting {len(tensors)} tensors...", flush=True)
    # Mutate the SAME dict in place (not a second `converted` dict) --
    # each old fp32 tensor is freed as soon as its key is overwritten,
    # instead of keeping two full copies alive until the end.
    for name in list(tensors.keys()):
        tensor = tensors[name]
        if tensor.is_floating_point():
            tensors[name] = tensor.to(torch.float16)
        # else: leave integer/bool tensors untouched, in place already.

    print(f"  Saving {os.path.basename(path)}...", flush=True)
    save_file(tensors, path)

    del tensors
    gc.collect()

    after_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Done: {os.path.basename(path)}: {before_mb:.1f} MB -> {after_mb:.1f} MB", flush=True)


def main():
    print(f"Converting weights in {WEIGHTS_DIR} to fp16...", flush=True)
    for fname in FILES_TO_CONVERT:
        path = os.path.join(WEIGHTS_DIR, fname)
        if not os.path.exists(path):
            print(f"  WARNING: {fname} not found at {path}, skipping.", flush=True)
            continue
        try:
            convert_file(path)
        except Exception:
            print(f"  FAILED converting {fname} -- traceback follows:", flush=True)
            traceback.print_exc()
            sys.exit(1)
    print("Done. Re-run package_submission.py to rebuild the zip with the smaller weights.", flush=True)


if __name__ == "__main__":
    main()