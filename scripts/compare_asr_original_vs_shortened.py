from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import librosa
import soundfile as sf
import torch
from jiwer import cer
from opencc import OpenCC
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

WHISPER_MODEL_NAME = "openai/whisper-small"
WAV2VEC2_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"

cc = OpenCC("t2s")


# -------------------------
# Text normalization
# -------------------------
def normalize(text: str) -> str:
    text = text.strip()
    text = cc.convert(text)
    text = re.sub(r"\s+", "", text)
    return text


# -------------------------
# Audio loader
# -------------------------
def load_audio(path: str, target_sr: int = 16000):
    audio, sr = sf.read(path)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return torch.tensor(audio, dtype=torch.float32)


# -------------------------
# Error analysis
# -------------------------
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


# -------------------------
# Whisper
# -------------------------
def whisper_transcribe(model, processor, device, audio_path: str) -> str:
    waveform = load_audio(audio_path)

    inputs = processor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    )

    with torch.no_grad():
        pred_ids = model.generate(
            inputs["input_features"].to(device),
            language="zh",
            task="transcribe",
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            max_new_tokens=128,
            do_sample=False,
        )

    return processor.batch_decode(pred_ids, skip_special_tokens=True)[0]


# -------------------------
# wav2vec2
# -------------------------
def wav2vec2_transcribe(model, processor, device, audio_path: str) -> str:
    waveform = load_audio(audio_path)

    inputs = processor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits

    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


# -------------------------
# Evaluate one model on one CSV
# -------------------------
def evaluate_dataset(model_name: str, input_csv: str, output_pred_csv: str, device: str):
    refs = []
    hyps = []
    rows_out = []
    total_counts = {"substitution": 0, "insertion": 0, "deletion": 0}
    total_chars = 0

    if model_name == "whisper":
        processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
        model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_NAME).to(device)
        transcribe_fn = lambda path: whisper_transcribe(model, processor, device, path)

    elif model_name == "wav2vec2":
        processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_NAME)
        model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_MODEL_NAME).to(device)
        transcribe_fn = lambda path: wav2vec2_transcribe(model, processor, device, path)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\n=== Running {model_name} on {input_csv} ({len(rows)} clips) ===")

    for i, row in enumerate(rows, 1):
        audio = row["audio"].strip()
        ref = normalize(row["text"])

        print(f"[{i}/{len(rows)}] {audio}")

        hyp_raw = transcribe_fn(audio)
        hyp = normalize(hyp_raw)

        refs.append(ref)
        hyps.append(hyp)

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

    print(f"Saved predictions: {output_path}")
    print(f"CER: {result['cer']:.4f}")
    print(
        f"Sub={result['substitution']} "
        f"Ins={result['insertion']} "
        f"Del={result['deletion']}"
    )

    return result


# -------------------------
# Save combined outputs
# -------------------------
def save_summary(all_results: list[dict], output_json: str, output_csv: str):
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

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


def print_table(results: list[dict]):
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


# -------------------------
# Main
# -------------------------
def main():
    if len(sys.argv) != 5:
        print("Usage:")
        print(
            "python scripts/compare_asr_original_vs_shortened.py "
            "<original_input_csv> <shortened_input_csv> <output_summary_json> <output_summary_csv>"
        )
        sys.exit(1)

    original_csv = sys.argv[1]
    shortened_csv = sys.argv[2]
    output_summary_json = sys.argv[3]
    output_summary_csv = sys.argv[4]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = []

    runs = [
        ("whisper", "original", original_csv, "results/predictions/whisper_original_auto.csv"),
        ("whisper", "shortened", shortened_csv, "results/predictions/whisper_shortened_auto.csv"),
        ("wav2vec2", "original", original_csv, "results/predictions/wav2vec2_original_auto.csv"),
        ("wav2vec2", "shortened", shortened_csv, "results/predictions/wav2vec2_shortened_auto.csv"),
    ]

    for model_name, condition, input_csv, pred_csv in runs:
        result = evaluate_dataset(model_name, input_csv, pred_csv, device)
        result["condition"] = condition
        results.append(result)

    print_table(results)
    save_summary(results, output_summary_json, output_summary_csv)


if __name__ == "__main__":
    main()