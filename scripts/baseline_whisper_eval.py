from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import torch
import soundfile as sf
import librosa
from jiwer import cer
from opencc import OpenCC
from transformers import WhisperProcessor, WhisperForConditionalGeneration


MODEL_NAME = "openai/whisper-small"

cc = OpenCC("t2s")


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
# Normalize Chinese
# -------------------------
def normalize(text: str):
    text = text.strip()
    text = cc.convert(text)         # traditional → simplified
    text = re.sub(r"\s+", "", text)
    return text


# -------------------------
# Transcription
# -------------------------
def transcribe(model, processor, audio_path: str):
    waveform = load_audio(audio_path)

    inputs = processor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    )

    with torch.no_grad():
        pred_ids = model.generate(
            inputs["input_features"].to(model.device),
            language="zh",
            task="transcribe",
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            max_new_tokens=128,
            do_sample=False,
        )

    text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    return text


# -------------------------
# Evaluation
# -------------------------
def evaluate(input_csv: str, output_pred_csv: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    refs = []
    hyps = []
    rows_out = []

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} samples from {input_csv}")

    for i, row in enumerate(rows, 1):
        audio = row["audio"]
        ref = normalize(row["text"])

        print(f"\n[{i}/{len(rows)}] {audio}")

        hyp_raw = transcribe(model, processor, audio)
        hyp = normalize(hyp_raw)

        print("REF:", ref)
        print("HYP:", hyp)

        refs.append(ref)
        hyps.append(hyp)

        rows_out.append({
            "audio": audio,
            "reference": ref,
            "hypothesis": hyp,
        })

    score = cer(refs, hyps)
    print(f"\nCER: {score:.4f}")

    # save output
    output_path = Path(output_pred_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "reference", "hypothesis"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Saved: {output_path}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:")
        print("python baseline_whisper_eval.py <input_csv> <output_pred_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_pred_csv = sys.argv[2]

    evaluate(input_csv, output_pred_csv)