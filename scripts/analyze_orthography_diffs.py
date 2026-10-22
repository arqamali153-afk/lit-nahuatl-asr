"""
analyze_orthography_diffs.py

Run this locally. Compares original_sentence vs sentence column-by-column
using a character-level diff and tallies which substitutions occur most
often. Output is an AGGREGATE frequency table (counts of character
changes, e.g. "h -> j : 342 times") -- it contains no actual sentence
content, so it's safe to share back for building the real regex rules.

Usage:
    python analyze_orthography_diffs.py data/nahuatl_dev/metadata.tsv
    python analyze_orthography_diffs.py path/to/train_metadata.csv --sep=,
"""

import argparse
import difflib
from collections import Counter
import pandas as pd


def char_substitutions(a: str, b: str):
    sm = difflib.SequenceMatcher(None, a, b)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            a_span, b_span = a[i1:i2], b[j1:j2]
            for x, y in zip(a_span, b_span):
                yield ("sub", x, y)
            if len(a_span) > len(b_span):
                for x in a_span[len(b_span):]:
                    yield ("del", x, None)
            elif len(b_span) > len(a_span):
                for y in b_span[len(a_span):]:
                    yield ("ins", None, y)
        elif tag == "delete":
            for x in a[i1:i2]:
                yield ("del", x, None)
        elif tag == "insert":
            for y in b[j1:j2]:
                yield ("ins", None, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_path")
    parser.add_argument("--sep", default="\t")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_path, sep=args.sep)
    if "original_sentence" not in df.columns or "sentence" not in df.columns:
        print("Expected columns not found. Available columns:", list(df.columns))
        return

    counter = Counter()
    total_rows = 0
    changed_rows = 0

    for orig, norm in zip(df["original_sentence"].fillna(""), df["sentence"].fillna("")):
        total_rows += 1
        if orig == norm:
            continue
        changed_rows += 1
        for kind, x, y in char_substitutions(orig, norm):
            counter[(kind, x, y)] += 1

    print(f"Total rows: {total_rows}, rows with any difference: {changed_rows}\n")
    print("Top 30 character-level changes (kind, from, to): count")
    for (kind, x, y), count in counter.most_common(30):
        print(f"  {kind:5s} {x!r:>5} -> {y!r:<5} : {count}")


if __name__ == "__main__":
    main()
