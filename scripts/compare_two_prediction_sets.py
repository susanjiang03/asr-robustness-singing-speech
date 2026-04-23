from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from jiwer import cer


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

                if best == sub:
                    back[i][j] = "S"
                elif best == ins:
                    back[i][j] = "I"
                else:
                    back[i][j] = "D"

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


def get_text_columns(row: dict[str, str]):
    if "reference" in row and "hypothesis" in row:
        return row["reference"], row["hypothesis"]

    if "reference_normalized" in row and "hypothesis_normalized" in row:
        return row["reference_normalized"], row["hypothesis_normalized"]

    raise ValueError(
        "CSV must contain either "
        "('reference','hypothesis') or "
        "('reference_normalized','hypothesis_normalized') columns."
    )


def evaluate_prediction_csv(input_csv: str):
    refs = []
    hyps = []
    total_counts = {"substitution": 0, "insertion": 0, "deletion": 0}
    total_chars = 0
    total_rows = 0

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            ref, hyp = get_text_columns(row)
            ref = ref.strip()
            hyp = hyp.strip()

            counts = levenshtein_ops(ref, hyp)

            refs.append(ref)
            hyps.append(hyp)

            for k in total_counts:
                total_counts[k] += counts[k]

            total_chars += len(ref)
            total_rows += 1

    result = {
        "input_csv": input_csv,
        "rows": total_rows,
        "total_reference_characters": total_chars,
        "cer": cer(refs, hyps),
        "substitution": total_counts["substitution"],
        "insertion": total_counts["insertion"],
        "deletion": total_counts["deletion"],
        "substitution_rate": total_counts["substitution"] / total_chars if total_chars else 0.0,
        "insertion_rate": total_counts["insertion"] / total_chars if total_chars else 0.0,
        "deletion_rate": total_counts["deletion"] / total_chars if total_chars else 0.0,
    }

    return result


def print_comparison(label_a: str, a: dict, label_b: str, b: dict):
    print("\n=== Comparison ===")
    print(f"{'Metric':<20} {label_a:<18} {label_b:<18} {'Delta(B-A)':<18}")
    print("-" * 74)

    metrics = [
        ("cer", "CER"),
        ("substitution", "Substitution"),
        ("insertion", "Insertion"),
        ("deletion", "Deletion"),
        ("substitution_rate", "Substitution Rate"),
        ("insertion_rate", "Insertion Rate"),
        ("deletion_rate", "Deletion Rate"),
    ]

    for key, label in metrics:
        va = a[key]
        vb = b[key]
        delta = vb - va

        if isinstance(va, float):
            print(f"{label:<20} {va:<18.4f} {vb:<18.4f} {delta:<18.4f}")
        else:
            print(f"{label:<20} {va:<18} {vb:<18} {delta:<18}")


def save_outputs(label_a: str, a: dict, label_b: str, b: dict, output_json: str, output_csv: str):
    output_json_path = Path(output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    combined = {
        "system_a": {"label": label_a, **a},
        "system_b": {"label": label_b, **b},
        "delta_b_minus_a": {
            "cer": b["cer"] - a["cer"],
            "substitution": b["substitution"] - a["substitution"],
            "insertion": b["insertion"] - a["insertion"],
            "deletion": b["deletion"] - a["deletion"],
            "substitution_rate": b["substitution_rate"] - a["substitution_rate"],
            "insertion_rate": b["insertion_rate"] - a["insertion_rate"],
            "deletion_rate": b["deletion_rate"] - a["deletion_rate"],
        },
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", label_a, label_b, "delta_b_minus_a"])
        writer.writerow(["cer", f"{a['cer']:.4f}", f"{b['cer']:.4f}", f"{b['cer'] - a['cer']:.4f}"])
        writer.writerow(["substitution", a["substitution"], b["substitution"], b["substitution"] - a["substitution"]])
        writer.writerow(["insertion", a["insertion"], b["insertion"], b["insertion"] - a["insertion"]])
        writer.writerow(["deletion", a["deletion"], b["deletion"], b["deletion"] - a["deletion"]])
        writer.writerow(["substitution_rate", f"{a['substitution_rate']:.4f}", f"{b['substitution_rate']:.4f}", f"{b['substitution_rate'] - a['substitution_rate']:.4f}"])
        writer.writerow(["insertion_rate", f"{a['insertion_rate']:.4f}", f"{b['insertion_rate']:.4f}", f"{b['insertion_rate'] - a['insertion_rate']:.4f}"])
        writer.writerow(["deletion_rate", f"{a['deletion_rate']:.4f}", f"{b['deletion_rate']:.4f}", f"{b['deletion_rate'] - a['deletion_rate']:.4f}"])

    print(f"\nSaved JSON: {output_json_path}")
    print(f"Saved CSV : {output_csv_path}")


def main():
    if len(sys.argv) != 7:
        print("Usage:")
        print(
            "python scripts/compare_two_prediction_sets.py "
            "<label_a> <predictions_csv_a> <label_b> <predictions_csv_b> "
            "<output_json> <output_csv>"
        )
        sys.exit(1)

    label_a = sys.argv[1]
    pred_a = sys.argv[2]
    label_b = sys.argv[3]
    pred_b = sys.argv[4]
    output_json = sys.argv[5]
    output_csv = sys.argv[6]

    result_a = evaluate_prediction_csv(pred_a)
    result_b = evaluate_prediction_csv(pred_b)

    print_comparison(label_a, result_a, label_b, result_b)
    save_outputs(label_a, result_a, label_b, result_b, output_json, output_csv)


if __name__ == "__main__":
    main()