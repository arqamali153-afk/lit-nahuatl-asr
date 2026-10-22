"""
download_mms_weights.py

Run this ONCE, locally, on a machine WITH internet access -- NEVER inside
the competition's air-gapped container. Downloads facebook/mms-1b-all
into src/weights/mms-1b-all/, filtered to skip the ~1100+ adapter heads
for languages this track doesn't need.

Size note: the shared backbone (model.safetensors, ~1B parameters) is
unavoidable and dominates the download regardless of language count.
Each individual adapter is only ~2-2.5M parameters (a few MB), so
filtering saves a modest amount -- not gigabytes. Don't expect this to
dramatically shrink your zip; the backbone is the real size driver.

Usage:
    pip install huggingface_hub
    python download_mms_weights.py
"""

import os
from huggingface_hub import snapshot_download

REPO_ID = "facebook/mms-1b-all"
TARGET_LANGUAGES = ["nhi", "azz", "nhw", "spa"]
OUTPUT_DIR = os.path.join("src", "weights", "mms-1b-all")


def main():
    allow_patterns = [
        # Base model + shared config/tokenizer files -- required
        # regardless of which languages you use.
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "special_tokens_map.json",
        "model.safetensors",
        # Only the adapter heads this competition track actually needs.
        *[f"adapter.{lang}.safetensors" for lang in TARGET_LANGUAGES],
    ]

    local_path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=OUTPUT_DIR,
        allow_patterns=allow_patterns,
    )

    print(f"Downloaded to: {local_path}")
    print("Contents:")
    total_mb = 0.0
    for root, _dirs, files in os.walk(local_path):
        for f in files:
            full = os.path.join(root, f)
            size_mb = os.path.getsize(full) / (1024 * 1024)
            total_mb += size_mb
            print(f"  {os.path.relpath(full, local_path)}  ({size_mb:.1f} MB)")
    print(f"Total: {total_mb:.1f} MB")

    # Sanity check: confirm every requested adapter actually exists in the
    # repo under these exact codes. If any are missing, the ISO codes in
    # asr_model.py's LANGUAGE_TO_ADAPTER mapping need to be corrected
    # BEFORE you trust the routing logic.
    missing = [
        lang for lang in TARGET_LANGUAGES
        if not os.path.exists(os.path.join(local_path, f"adapter.{lang}.safetensors"))
    ]
    if missing:
        print(
            f"\nWARNING: no adapter file found for language code(s): {missing}. "
            f"Check the actual adapter list in the {REPO_ID} repo on Hugging Face "
            f"and correct LANGUAGE_TO_ADAPTER in asr_model.py before relying on "
            f"dialect routing."
        )
    else:
        print(f"\nAll {len(TARGET_LANGUAGES)} requested adapters found and downloaded.")


if __name__ == "__main__":
    main()