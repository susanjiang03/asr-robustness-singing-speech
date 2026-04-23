from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import librosa
import soundfile as sf
import torch
from jiwer import cer
from opencc import OpenCC
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE_MODEL_NAME = "openai/whisper-small"
TARGET_SR = 16000
cc = OpenCC("t2s")


def normalize(text: str) -> str:
    text = text.strip()
    text = cc.convert(text)
    text = re.sub(r"\s+", "", text)
    return text


def load_audio(path: str, target_sr: int = TARGET_SR):
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    return torch.tensor(audio, dtype=torch.float32)


def evaluate(input_csv: str, adapter_dir: str, output_csv: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = WhisperProcessor.from_pretrained(adapter_dir)

    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)

    refs = []
    hyps = []
    rows_out = []

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from {input_csv}")

    for i, row in enumerate(rows, 1):
        audio_path = row["audio"].strip()
        ref = normalize(row["text"])

        waveform = load_audio(audio_path)

        inputs = processor(
            waveform.numpy(),
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )

        with torch.no_grad():
            pred_ids = model.generate(
                inputs["input_features"].to(device),

                # language
                language="zh",
                task="transcribe",

                # 🔥 anti-repetition (MOST IMPORTANT)
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,

                # 🔥 control output length
                max_new_tokens=100,
                length_penalty=0.8,

                # 🔥 better decoding
                num_beams=3,
                do_sample=False,
            )

        hyp = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        hyp = re.sub(r"(.)\1{2,}", r"\1", hyp)
        hyp = normalize(hyp)

        refs.append(ref)
        hyps.append(hyp)

        rows_out.append({
            "audio": audio_path,
            "reference": ref,
            "hypothesis": hyp,
        })

        print(f"[{i}/{len(rows)}] REF={ref}")
        print(f"           HYP={hyp}")

    score = cer(refs, hyps)
    print(f"\nCER: {score:.4f}")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "reference", "hypothesis"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Saved predictions: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage:")
        print("python scripts/eval_whisper_lora.py <input_csv> <adapter_dir> <output_csv>")
        sys.exit(1)

    evaluate(sys.argv[1], sys.argv[2], sys.argv[3])