"""
Evaluation utilities for ASR models
"""
import csv
import json
from pathlib import Path

import torch
from jiwer import cer

from ..models.asr_models import get_model
from ..utils.text_utils import levenshtein_ops, normalize_text


def evaluate_dataset(
    model_name: str,
    input_csv: str,
    output_pred_csv: str,
    device: str = "cpu",
    vosk_model_path: str = None,
    verbose: bool = True
):
    """Evaluate a single ASR model on a dataset"""
    
    refs = []
    hyps = []
    rows_out = []
    total_counts = {"substitution": 0, "insertion": 0, "deletion": 0}
    total_chars = 0

    # Initialize model
    model = get_model(model_name, device, vosk_model_path)

    # Load dataset
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if verbose:
        print(f"\n=== Running {model_name} on {input_csv} ({len(rows)} clips) ===")

    for i, row in enumerate(rows, 1):
        audio = row["audio"].strip()
        ref = normalize_text(row["text"])

        if verbose:
            print(f"[{i}/{len(rows)}] {audio}")

        # Transcribe
        hyp_raw = model.transcribe(audio)
        hyp = normalize_text(hyp_raw)

        refs.append(ref)
        hyps.append(hyp)

        # Calculate error operations
        counts = levenshtein_ops(ref, hyp)
        for k in total_counts:
            total_counts[k] += counts[k]
        total_chars += len(ref)

        rows_out.append({
            "audio": audio,
            "reference": ref,
            "hypothesis": hyp,
            "substitution": counts["substitution"],
            "insertion": counts["insertion"],
            "deletion": counts["deletion"],
        })

    # Calculate final metrics
    result = {
        "model": model_name,
        "input_csv": input_csv,
        "rows": len(rows),
        "total_reference_characters": total_chars,
        "cer": cer(refs, hyps),
        "substitution": total_counts["substitution"],
        "insertion": total_counts["insertion"],
        "deletion": total_counts["deletion"],
        "substitution_rate": total_counts["substitution"] / total_chars if total_chars else 0.0,
        "insertion_rate": total_counts["insertion"] / total_chars if total_chars else 0.0,
        "deletion_rate": total_counts["deletion"] / total_chars if total_chars else 0.0,
    }

    # Save predictions
    output_path = Path(output_pred_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "audio",
                "reference",
                "hypothesis",
                "substitution",
                "insertion",
                "deletion",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    if verbose:
        print(f"Saved predictions: {output_path}")
        print(f"CER: {result['cer']:.4f}")
        print(f"Sub={result['substitution']} Ins={result['insertion']} Del={result['deletion']}")

    return result


def save_evaluation_summary(all_results: list[dict], output_json: str, output_csv: str):
    """Save evaluation results to JSON and CSV files"""
    
    # Save JSON
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Save CSV
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "condition",
            "input_csv",
            "rows",
            "total_reference_characters",
            "cer",
            "substitution",
            "insertion",
            "deletion",
            "substitution_rate",
            "insertion_rate",
            "deletion_rate",
        ])

        for r in all_results:
            writer.writerow([
                r["model"],
                r["condition"],
                r["input_csv"],
                r["rows"],
                r["total_reference_characters"],
                f"{r['cer']:.4f}",
                r["substitution"],
                r["insertion"],
                r["deletion"],
                f"{r['substitution_rate']:.4f}",
                f"{r['insertion_rate']:.4f}",
                f"{r['deletion_rate']:.4f}",
            ])

    print(f"\nSaved summary JSON: {json_path}")
    print(f"Saved summary CSV : {csv_path}")


def print_results_table(results: list[dict]):
    """Print formatted results table"""
    print("\n=== Final Comparison ===")
    print(
        f"{'Model':<12} {'Condition':<12} {'CER':<10} "
        f"{'Sub':<8} {'Ins':<8} {'Del':<8} "
        f"{'SubRate':<10} {'InsRate':<10} {'DelRate':<10}"
    )
    print("-" * 100)

    for r in results:
        print(
            f"{r['model']:<12} {r['condition']:<12} {r['cer']:<10.4f} "
            f"{r['substitution']:<8} {r['insertion']:<8} {r['deletion']:<8} "
            f"{r['substitution_rate']:<10.4f} {r['insertion_rate']:<10.4f} {r['deletion_rate']:<10.4f}"
        )
