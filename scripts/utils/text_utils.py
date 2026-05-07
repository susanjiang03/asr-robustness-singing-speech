"""
Text processing utilities for ASR robustness project
"""
import re
from opencc import OpenCC

# Initialize OpenCC converter for Traditional to Simplified Chinese
cc = OpenCC("t2s")


def normalize_text(text: str) -> str:
    """Normalize text for ASR evaluation"""
    text = text.strip()
    text = cc.convert(text)  # Traditional to Simplified
    text = re.sub(r"\s+", "", text)  # Remove all whitespace
    return text


def levenshtein_ops(ref: str, hyp: str):
    """Calculate Levenshtein operations for detailed error analysis"""
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
