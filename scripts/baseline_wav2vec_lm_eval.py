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
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from pyctcdecode import build_ctcdecoder


MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
TARGET_SR = 16000
cc = OpenCC("t2s")


def normalize(text: str) -> str:
    text = text.strip()
    text = cc.convert(text)
    text = re.sub(r"\s+", "", text)
    return text


def load_audio(path: str, target_sr: int = TARGET_SR):
    audio, sr = sf.read(path)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return torch.tensor(audio, dtype=torch.float32)


def load_lm_corpus_text(corpus_path: str) -> list[str]:
    lines = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = normalize(line)
            if line:
                lines.append(line)
    return lines


def transcribe_with_lm(model, processor, decoder, audio_path: str, device: str):
    waveform = load_audio(audio_path)

    inputs = processor(
        waveform.numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits

    # logits shape: [1, time, vocab]
    logits_np = logits[0].cpu().numpy()

    text = decoder.decode(logits_np)
    return text


def evaluate(input_csv: str, output_csv: str, lm_corpus_txt: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device)

    vocab_dict = processor.tokenizer.get_vocab()
    # sort by token id
    sorted_vocab = [token for token, idx in sorted(vocab_dict.items(), key=lambda x: x[1])]

    # pyctcdecode decoder
    decoder = build_ctcdecoder(
        labels=sorted_vocab,
    )

    # Optional: if you later build a KenLM binary/arpa, pass kenlm_model_path=...
    # For now this creates a CTC decoder scaffold; you can later extend with a real LM file.

    refs = []
    hyps = []
    rows_out = []

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} samples from {input_csv}")

    # Read corpus just so workflow is explicit and paper-reproducible
    lm_lines = load_lm_corpus_text(lm_corpus_txt)
    print(f"Loaded {len(lm_lines)} LM text lines from {lm_corpus_txt}")

    for i, row in enumerate(rows, 1):
        audio = row["audio"].strip()
        ref = normalize(row["text"])

        print(f"\n[{i}/{len(rows)}] {audio}")

        hyp = transcribe_with_lm(model, processor, decoder, audio, device)
        hyp = normalize(hyp)

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


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage:")
        print("python scripts/baseline_wav2vec_lm_eval.py <input_csv> <output_csv> <lm_corpus_txt>")
        sys.exit(1)

    evaluate(sys.argv[1], sys.argv[2], sys.argv[3])