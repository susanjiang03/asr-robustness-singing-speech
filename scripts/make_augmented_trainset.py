from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


TARGET_SR = 16000

# Safe augmentations for singing ASR
AUGMENTATIONS = [
    ("orig", []),

    ("pitch_p1", [("pitch", 1.0)]),
    ("pitch_m1", [("pitch", -1.0)]),
    ("pitch_p2", [("pitch", 2.0)]),
    ("pitch_m2", [("pitch", -2.0)]),

    ("stretch_095", [("stretch", 0.95)]),
    ("stretch_105", [("stretch", 1.05)]),

    ("pitch_p1_stretch_095", [("pitch", 1.0), ("stretch", 0.95)]),
    ("pitch_m1_stretch_105", [("pitch", -1.0), ("stretch", 1.05)]),
]


def load_audio(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    audio, sr = sf.read(path)

    # stereo -> mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio.astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    return audio.astype(np.float32)


def apply_one(audio: np.ndarray, sr: int, transform: tuple[str, float]) -> np.ndarray:
    kind, value = transform

    if kind == "pitch":
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=value)

    if kind == "stretch":
        return librosa.effects.time_stretch(audio, rate=value)

    raise ValueError(f"Unknown transform: {kind}")


def apply_pipeline(audio: np.ndarray, sr: int, transforms: list[tuple[str, float]]) -> np.ndarray:
    out = audio.copy()
    for t in transforms:
        out = apply_one(out, sr, t)
    return normalize_audio(out)


def main(input_csv: str, output_dir: str, output_csv: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rows_out: list[dict[str, str]] = []

    for idx, row in enumerate(rows, 1):
        input_audio = row["audio"].strip()
        text = row["text"].strip()

        audio = load_audio(input_audio, TARGET_SR)
        base = Path(input_audio).stem

        print(f"\n[{idx}/{len(rows)}] Processing: {input_audio}")

        for aug_name, transforms in AUGMENTATIONS:
            audio_aug = apply_pipeline(audio, TARGET_SR, transforms)

            out_name = f"{base}_{aug_name}.wav"
            out_path = os.path.join(output_dir, out_name)

            sf.write(out_path, audio_aug, TARGET_SR)

            rows_out.append({
                "audio": out_path,
                "text": text,
            })

            print(f"  saved: {out_path}")

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "text"])
        writer.writeheader()
        writer.writerows(rows_out)

    print("\n=== Done ===")
    print(f"Input rows            : {len(rows)}")
    print(f"Augmentations per row : {len(AUGMENTATIONS)}")
    print(f"Output rows           : {len(rows_out)}")
    print(f"Output audio dir      : {output_dir}")
    print(f"Output CSV            : {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage:")
        print("python scripts/make_augmented_trainset.py <input_csv> <output_dir> <output_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_dir = sys.argv[2]
    output_csv = sys.argv[3]

    main(input_csv, output_dir, output_csv)