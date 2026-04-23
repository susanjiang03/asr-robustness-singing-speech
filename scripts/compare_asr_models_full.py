from __future__ import annotations

import csv
import re
import torch
import soundfile as sf
import librosa
from jiwer import cer
from opencc import OpenCC

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)


cc = OpenCC("t2s")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Audio loader
# -------------------------
def load_audio(path: str, target_sr: int = 16000):
    audio, sr = sf.read(path)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio


# -------------------------
# Text normalization
# -------------------------
def normalize(text: str):
    text = text.strip()
    text = cc.convert(text)
    text = re.sub(r"\s+", "", text)
    return text


# -------------------------
# Whisper
# -------------------------
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small"
).to(DEVICE)


def whisper_transcribe(path):
    audio = load_audio(path)

    inputs = whisper_processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    )

    with torch.no_grad():
        pred_ids = whisper_model.generate(
            inputs["input_features"].to(DEVICE),
            language="zh",
            task="transcribe",
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            max_new_tokens=128,
            do_sample=False,
        )

    return whisper_processor.batch_decode(pred_ids, skip_special_tokens=True)[0]


# -------------------------
# wav2vec2
# -------------------------
w2v_processor = Wav2Vec2Processor.from_pretrained(
    "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
)
w2v_model = Wav2Vec2ForCTC.from_pretrained(
    "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
).to(DEVICE)


def wav2vec_transcribe(path):
    audio = load_audio(path)

    inputs = w2v_processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        logits = w2v_model(inputs.input_values.to(DEVICE)).logits

    pred_ids = torch.argmax(logits, dim=-1)
    return w2v_processor.batch_decode(pred_ids)[0]


# -------------------------
# Evaluation
# -------------------------
def evaluate(csv_path):
    refs = []
    whisper_hyps = []
    w2v_hyps = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        rows = list(reader)

    print(f"\nEvaluating {len(rows)} samples from {csv_path}")

    for i, row in enumerate(rows, 1):
        audio = row["audio"]
        ref = normalize(row["text"])

        print(f"\n[{i}] {audio}")

        whisper_text = normalize(whisper_transcribe(audio))
        w2v_text = normalize(wav2vec_transcribe(audio))

        print("REF:", ref)
        print("Whisper:", whisper_text)
        print("wav2vec2:", w2v_text)

        refs.append(ref)
        whisper_hyps.append(whisper_text)
        w2v_hyps.append(w2v_text)

    print("\n===== FINAL RESULTS =====")
    print("Whisper CER :", cer(refs, whisper_hyps))
    print("wav2vec2 CER:", cer(refs, w2v_hyps))


if __name__ == "__main__":
    evaluate("data/test_shortened_0.csv")