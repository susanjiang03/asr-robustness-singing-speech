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
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
cc = OpenCC("t2s")


# -------------------------
# Audio loader
# -------------------------
def load_audio(path: str, target_sr: int = 16000):
    audio, sr = sf.read(path)

    # stereo -> mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # resample
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return torch.tensor(audio, dtype=torch.float32)


# -------------------------
# Normalize Chinese
# -------------------------
def normalize(text: str):
    text = text.strip()
    text = cc.convert(text)  # traditional -> simplified
    text = re.sub(r"\s+", "", text)
    return text


# -------------------------
# Transcribe (CTC decoding)
# -------------------------
def transcribe(model, processor, audio_path: str):
    waveform = load_audio(audio_path)

    inputs = processor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        logits = model(inputs.input_values.to(model.device)).logits

    pred_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(pred_ids)[0]

    return text


# -------------------------
# Evaluate
# -------------------------
def evaluate(input_csv: str, output_csv: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device)

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

    # save predictions
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["audio", "reference", "hypothesis"]
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Saved: {output_path}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:")
        print("python scripts/baseline_wav2vec_eval.py <input_csv> <output_csv>")
        sys.exit(1)

    evaluate(sys.argv[1], sys.argv[2])