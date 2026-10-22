"""
prepare_lm_corpus.py (v2 -- Spanish-inclusive)

Builds:
    1. data/unigrams.txt       -> vocabulary list for pyctcdecode beam search
    2. data/nahuatl_corpus.txt -> text corpus for KenLM n-gram training

v1 trained purely on the Nahuatl `sentence` column and regressed on the
real (espnah-dominated) test set: Spanish words in genuinely code-switched
audio looked like out-of-vocabulary hallucinations to a Nahuatl-only
lexicon/LM, and got penalized or reshaped into nonsense.

v2 adds two more in-domain sources, both already unused in your data:
  - `spanish_sentence` column in corpus.tsv: parallel Spanish text,
    broader generic Spanish vocabulary coverage than what happens to
    occur in the (small) Nahuatl transcripts alone.
  - `lid_tokens` column: word-level language tags (e.g. "quenin//nhi
    tipanotoc//nhi"). Words tagged //spa or //adapted-spanish-loan are
    extracted directly -- the highest-confidence, most contextually
    relevant Spanish vocabulary available, since these are real
    documented code-switches, not just generic Spanish text.

Note: Spanish spelling itself is unaffected by the Nahuatl colonial-vs-
modern orthography question -- no orthography-space concern applies to
the Spanish portions added here.
"""

import os
import re
import pandas as pd

TETELANCINGO_CORPUS = "data/tetelancingo_nahuatl/corpus.tsv"
OUTPUT_CORPUS = "data/nahuatl_corpus.txt"
OUTPUT_UNIGRAMS = "data/unigrams.txt"


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"//[a-z\-]+", "", text)   # strip //nhi, //spa, //adapted-spanish-loan tags
    text = re.sub(r"\[.*?\]", "", text)       # strip bracketed notes
    # Keep letters, apostrophe, accented vowels, ñ (needed for both
    # Nahuatl diacritics and standard Spanish).
    text = re.sub(r"[^a-zñ 'áéíóú]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_lid_tagged_words(lid_tokens_str: str, wanted_tags: set) -> list:
    """Parses 'word//tag word//tag ...' and returns cleaned words whose
    tag is in wanted_tags. Defensive: skips malformed tokens rather than
    raising, since real-world annotation data is rarely 100% clean."""
    if not isinstance(lid_tokens_str, str):
        return []
    words = []
    for token in lid_tokens_str.split():
        if "//" not in token:
            continue
        word, _, tag = token.rpartition("//")
        if tag in wanted_tags:
            cleaned = clean_text(word)
            if cleaned:
                words.append(cleaned)
    return words


def main():
    if not os.path.exists(TETELANCINGO_CORPUS):
        print(f"ERROR: {TETELANCINGO_CORPUS} not found.")
        return

    df = pd.read_csv(TETELANCINGO_CORPUS, sep="\t")

    lines = []  # full corpus lines (for KenLM n-gram training)
    words = set()  # unigram lexicon (dedup fine here -- it's a whitelist, not a frequency table)

    # 1. Nahuatl (colonial-orthography) sentences -- same as v1, matches
    #    the raw model output's spelling convention. No dedup: frequency
    #    matters for the n-gram model.
    if "sentence" in df.columns:
        for s in df["sentence"].dropna():
            cleaned = clean_text(s)
            if cleaned:
                lines.append(cleaned)
                words.update(w for w in cleaned.split() if len(w) > 1)
        print(f"Added {len(df['sentence'].dropna())} lines from 'sentence' column (Nahuatl, colonial orthography).")
    else:
        print("WARNING: 'sentence' column not found -- skipping Nahuatl corpus lines.")

    # 2. Spanish parallel text -- broader generic Spanish vocabulary,
    #    not orthography-sensitive.
    if "spanish_sentence" in df.columns:
        spanish_rows = df["spanish_sentence"].dropna()
        for s in spanish_rows:
            cleaned = clean_text(s)
            if cleaned:
                lines.append(cleaned)
                words.update(w for w in cleaned.split() if len(w) > 1)
        print(f"Added {len(spanish_rows)} lines from 'spanish_sentence' column (Spanish).")
    else:
        print("NOTE: 'spanish_sentence' column not found -- skipping (check actual column name if this is unexpected).")

    # 3. lid_tokens-tagged Spanish/loanword vocabulary -- real, in-context
    #    code-switches. Adds to the unigram lexicon directly (highest
    #    confidence source); also folded into corpus lines as short
    #    "sentences" so KenLM sees them too, at low weight given they're
    #    single words, not natural sentences.
    if "lid_tokens" in df.columns:
        spa_word_count = 0
        for lid_str in df["lid_tokens"].dropna():
            spa_words = extract_lid_tagged_words(lid_str, {"spa", "adapted-spanish-loan"})
            for w in spa_words:
                words.add(w)
                lines.append(w)  # single-word line -- still valid unigram evidence for KenLM
                spa_word_count += 1
        print(f"Added {spa_word_count} lid_tokens-tagged Spanish/loanword occurrences.")
    else:
        print("NOTE: 'lid_tokens' column not found -- skipping (check actual column name if this is unexpected).")

    if not lines:
        print("ERROR: No corpus lines extracted from any source -- check column names above.")
        return

    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"\nSaved {len(lines)} total corpus lines to {OUTPUT_CORPUS}")

    with open(OUTPUT_UNIGRAMS, "w", encoding="utf-8") as f:
        for w in sorted(words):
            f.write(w + "\n")
    print(f"Saved {len(words)} unigram tokens to {OUTPUT_UNIGRAMS}")


if __name__ == "__main__":
    main()