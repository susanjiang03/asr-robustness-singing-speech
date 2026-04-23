from __future__ import annotations

import csv


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


def analyze(csv_path: str):
    total_counts = {"substitution": 0, "insertion": 0, "deletion": 0}
    total_chars = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            ref = row["reference"]
            hyp = row["hypothesis"]

            counts = levenshtein_ops(ref, hyp)

            for k in total_counts:
                total_counts[k] += counts[k]

            total_chars += len(ref)

    print("\n=== Error Type Analysis ===")
    print("Total characters:", total_chars)
    print("Substitution:", total_counts["substitution"])
    print("Insertion:", total_counts["insertion"])
    print("Deletion:", total_counts["deletion"])

    print("\n=== Rates ===")
    print("Substitution rate:", total_counts["substitution"] / total_chars)
    print("Insertion rate:", total_counts["insertion"] / total_chars)
    print("Deletion rate:", total_counts["deletion"] / total_chars)


if __name__ == "__main__":
    analyze("results/predictions/whisper_original.csv")
    analyze("results/predictions/whisper_shortened.csv")