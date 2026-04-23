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
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
TARGET_SR = 16000
cc = OpenCC("t2s")


# -------------------------
# Normalize Chinese
# -------------------------
def normalize(text: str) -> str:
    text = text.strip()
    text = cc.convert(text)
    text = re.sub(r"\s+", "", text)
    return text


# -------------------------
# Load audio
# -------------------------
def load_audio(path: str):
    audio, sr = sf.read(path)

    # stereo -> mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # resample
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    return torch.tensor(audio, dtype=torch.float32)


# -------------------------
# Build CTC decoder (beam only, no LM)
# -------------------------
def build_decoder(processor: Wav2Vec2Processor):
    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab = [token for token, idx in sorted(vocab_dict.items(), key=lambda x: x[1])]

    # normalize special tokens for pyctcdecode
    cleaned_vocab = []
    for token in sorted_vocab:
        if token == processor.tokenizer.pad_token:
            cleaned_vocab.append("")
        elif token == processor.tokenizer.word_delimiter_token:
            cleaned_vocab.append(" ")
        else:
            cleaned_vocab.append(token)

    decoder = build_ctcdecoder(labels=cleaned_vocab)
    return decoder


# -------------------------
# Beam-search transcription
# -------------------------
def transcribe(model, processor, decoder, audio_path: str, device: str):
    waveform = load_audio(audio_path)

    inputs = processor(
        waveform.numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits

    logits_np = logits[0].cpu().numpy()

    # beam search without external LM
    text = decoder.decode(logits_np, beam_width=10)
    return text


# -------------------------
# Evaluate
# -------------------------
def evaluate(input_csv: str, output_csv: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device)
    decoder = build_decoder(processor)

    refs = []
    hyps = []
    rows_out = []

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} samples from {input_csv}")

    for i, row in enumerate(rows, 1):
        audio = row["audio"].strip()
        ref = normalize(row["text"])

        print(f"\n[{i}/{len(rows)}] {audio}")

        hyp_raw = transcribe(model, processor, decoder, audio, device)
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

    output_path = Path(output_csv)
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
        print("python scripts/baseline_wav2vec_beam_eval.py <input_csv> <output_csv>")
        sys.exit(1)

    evaluate(sys.argv[1], sys.argv[2])