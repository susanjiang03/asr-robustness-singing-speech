from __future__ import annotations

import csv
from pathlib import Path
from jiwer import cer


def load_norm_pairs(path: str):
    refs = []
    hyps = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            refs.append(row["reference"])
            hyps.append(row["hypothesis"])

    return refs, hyps


def main():
    original_refs, original_hyps = load_norm_pairs(
        "results/predictions/whisper_original.csv"
    )
    shortened_refs, shortened_hyps = load_norm_pairs(
        "results/predictions/whisper_shortened.csv"
    )

    if len(original_refs) != len(shortened_refs):
        raise ValueError(
            f"Row count mismatch: original={len(original_refs)}, shortened={len(shortened_refs)}"
        )

    original_cer = cer(original_refs, original_hyps)
    shortened_cer = cer(shortened_refs, shortened_hyps)

    print("\n=== Whisper Comparison ===")
    print(f"Original CER : {original_cer:.4f}")
    print(f"Shortened CER: {shortened_cer:.4f}")
    print(f"Delta        : {shortened_cer - original_cer:+.4f}")

    output_path = Path("results/tables/original_vs_shortened.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "cer"])
        writer.writerow(["original", f"{original_cer:.4f}"])
        writer.writerow(["shortened", f"{shortened_cer:.4f}"])

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()