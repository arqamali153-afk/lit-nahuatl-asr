"""
model_soup_average_checkpoints.py

Averages the adapter+lm_head weights from two nearby-epoch checkpoints
(e.g. checkpoint-2350 [epoch 10] and checkpoint-2585 [epoch 11]) --
a cheap, low-risk way to smooth optimization noise, since both sit in
the same loss basin (consecutive epochs, similar eval WER).

Usage:
    python model_soup_average_checkpoints.py \
        /path/to/checkpoint-2350/model.safetensors \
        /path/to/checkpoint-2585/model.safetensors \
        /path/to/output/adapter.nhi.soup.safetensors
"""

import sys
from safetensors.torch import load_file, save_file

if len(sys.argv) != 4:
    print("Usage: python model_soup_average_checkpoints.py <ckpt1> <ckpt2> <output_path>")
    sys.exit(1)

ckpt1_path, ckpt2_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

print(f"Loading {ckpt1_path}...")
state1 = load_file(ckpt1_path)
print(f"Loading {ckpt2_path}...")
state2 = load_file(ckpt2_path)

relevant_keys = {k for k in state1.keys() if ("adapter" in k) or ("lm_head" in k)}
missing_in_2 = relevant_keys - set(state2.keys())
if missing_in_2:
    print(f"WARNING: {len(missing_in_2)} adapter/lm_head keys in checkpoint 1 "
          f"are missing from checkpoint 2 -- these will be taken from checkpoint 1 only, "
          f"not truly averaged. Check both checkpoints came from the same run.")

averaged_state = {}
for key in relevant_keys:
    t1 = state1[key]
    if key in state2:
        t2 = state2[key]
        averaged_state[key] = (t1.float() + t2.float()) / 2.0
    else:
        averaged_state[key] = t1

save_file(averaged_state, output_path)
print(f"\nSaved averaged (souped) adapter+lm_head weights to: {output_path}")
print(f"Averaged {len(averaged_state)} tensors.")
print("\nIMPORTANT: run eval_pooled_per_dialect.py against THIS file before assuming "
      "it's better than the single checkpoint-2585 extraction you already tested -- "
      "'same loss basin' is a reasonable assumption, not a guarantee, especially "
      "given nhw's validation number may already be somewhat inflated by low "
      "speaker diversity.")
