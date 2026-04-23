from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from jiwer import cer
from rapidfuzz import fuzz
from opencc import OpenCC


cc = OpenCC("t2s")


def normalize(text: str) -> str:
    text = text.strip()
    text = cc.convert(text)
    text = re.sub(r"\s+", "", text)
    return text


def load_dictionary(dict_path: str) -> list[str]:
    vocab = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            w = normalize(line)
            if w:
                vocab.append(w)

    # longer phrases first
    vocab = sorted(set(vocab), key=lambda x: (-len(x), x))
    return vocab


def build_candidate_windows(text: str, min_len: int = 2, max_len: int = 6):
    candidates = []
    n = len(text)
    for i in range(n):
        for j in range(i + min_len, min(n + 1, i + max_len + 1)):
            candidates.append((i, j, text[i:j]))
    return candidates


def correct_text_with_dictionary(
    text: str,
    vocab: list[str],
    threshold: int = 80,
    max_phrase_len: int = 6,
) -> str:
    """
    Greedy fuzzy replacement.
    Tries to replace short substrings with dictionary phrases if similarity is high.
    """
    text = normalize(text)
    if not text:
        return text

    changed = True
    max_rounds = 3
    round_count = 0

    while changed and round_count < max_rounds:
        changed = False
        round_count += 1

        windows = build_candidate_windows(text, min_len=2, max_len=max_phrase_len)

        best_match = None
        best_score = -1

        for start, end, sub in windows:
            for target in vocab:
                # only compare substrings with reasonably close length
                if abs(len(sub) - len(target)) > 2:
                    continue

                score = fuzz.ratio(sub, target)
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = (start, end, sub, target, score)

        if best_match is not None:
            start, end, sub, target, score = best_match
            if sub != target:
                text = text[:start] + target + text[end:]
                changed = True

    return text


def levenshtein_ops(ref: str, hyp: str):
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    back = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
        if i > 0:
            back[i][0] = "D"

    for j in range(n + 1):
        dp[0][j] = j
        if j > 0:
            back[0][j] = "I"

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "M"
            else:
                sub = dp[i - 1][j - 1] + 1
                ins = dp[i][j - 1] + 1
                dele = dp[i - 1][j] + 1

                best = min(sub, ins, dele)
                dp[i][j] = best
                back[i][j] = "S" if best == sub else ("I" if best == ins else "D")

    i, j = m, n
    counts = {"substitution": 0, "insertion": 0, "deletion": 0}

    while i > 0 or j > 0:
        op = back[i][j]
        if op == "M":
            i -= 1
            j -= 1
        elif op == "S":
            counts["substitution"] += 1
            i -= 1
            j -= 1
        elif op == "I":
            counts["insertion"] += 1
            j -= 1
        elif op == "D":
            counts["deletion"] += 1
            i -= 1
        else:
            break

    return counts


def summarize(refs: list[str], hyps: list[str]):
    total_counts = {"substitution": 0, "insertion": 0, "deletion": 0}
    total_chars = 0

    for ref, hyp in zip(refs, hyps):
        counts = levenshtein_ops(ref, hyp)
        for k in total_counts:
            total_counts[k] += counts[k]
        total_chars += len(ref)

    return {
        "cer": cer(refs, hyps),
        "total_reference_characters": total_chars,
        "substitution": total_counts["substitution"],
        "insertion": total_counts["insertion"],
        "deletion": total_counts["deletion"],
        "substitution_rate": total_counts["substitution"] / total_chars if total_chars else 0.0,
        "insertion_rate": total_counts["insertion"] / total_chars if total_chars else 0.0,
        "deletion_rate": total_counts["deletion"] / total_chars if total_chars else 0.0,
    }


def main(input_pred_csv: str, dict_txt: str, output_csv: str, threshold: int = 80):
    vocab = load_dictionary(dict_txt)
    print(f"Loaded {len(vocab)} dictionary entries from {dict_txt}")

    rows_out = []
    refs_before = []
    hyps_before = []
    refs_after = []
    hyps_after = []

    with open(input_pred_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        audio = row["audio"].strip()
        ref = normalize(row["reference"])
        hyp = normalize(row["hypothesis"])

        hyp_corrected = correct_text_with_dictionary(
            hyp,
            vocab,
            threshold=threshold,
            max_phrase_len=6,
        )

        refs_before.append(ref)
        hyps_before.append(hyp)

        refs_after.append(ref)
        hyps_after.append(hyp_corrected)

        rows_out.append({
            "audio": audio,
            "reference": ref,
            "hypothesis_before": hyp,
            "hypothesis_after": hyp_corrected,
        })

    before = summarize(refs_before, hyps_before)
    after = summarize(refs_after, hyps_after)

    print("\n=== Before Dictionary Correction ===")
    print(f"CER: {before['cer']:.4f}")
    print(f"Sub={before['substitution']} Ins={before['insertion']} Del={before['deletion']}")
    print(
        f"SubRate={before['substitution_rate']:.4f} "
        f"InsRate={before['insertion_rate']:.4f} "
        f"DelRate={before['deletion_rate']:.4f}"
    )

    print("\n=== After Dictionary Correction ===")
    print(f"CER: {after['cer']:.4f}")
    print(f"Sub={after['substitution']} Ins={after['insertion']} Del={after['deletion']}")
    print(
        f"SubRate={after['substitution_rate']:.4f} "
        f"InsRate={after['insertion_rate']:.4f} "
        f"DelRate={after['deletion_rate']:.4f}"
    )

    print(f"\nDelta CER: {after['cer'] - before['cer']:+.4f}")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["audio", "reference", "hypothesis_before", "hypothesis_after"],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Saved corrected predictions to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print("Usage:")
        print(
            "python scripts/apply_dictionary_correction.py "
            "<input_predictions_csv> <dictionary_txt> <output_csv> [threshold]"
        )
        sys.exit(1)

    input_pred_csv = sys.argv[1]
    dict_txt = sys.argv[2]
    output_csv = sys.argv[3]
    threshold = int(sys.argv[4]) if len(sys.argv) == 5 else 80

    main(input_pred_csv, dict_txt, output_csv, threshold)