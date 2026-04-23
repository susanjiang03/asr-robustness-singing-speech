from __future__ import annotations

import csv
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


INPUT_CSV = "data/test.csv"
OUTPUT_AUDIO_DIR = "data/clips/test_shortened"
OUTPUT_CSV = "data/test_shortened.csv"

TARGET_SR = 16000
HOP_LENGTH = 256
FRAME_LENGTH = 1024

# A voiced segment longer than this is treated as "elongated"
MIN_ELONGATED_SEC = 0.50

# Keep 75% of the original duration for elongated voiced regions
SHORTEN_FACTOR = 0.75

# Very short regions are not modified
MIN_SEGMENT_SAMPLES = 400


def load_audio(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    y, sr = sf.read(path)

    if len(y.shape) > 1:
        y = y.mean(axis=1)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    return y.astype(np.float32)


def find_voiced_segments(y: np.ndarray, sr: int):
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )

    voiced = ~np.isnan(f0)

    frame_segments = []
    start = None
    for i, is_voiced in enumerate(voiced):
        if is_voiced and start is None:
            start = i
        elif not is_voiced and start is not None:
            frame_segments.append((start, i))
            start = None

    if start is not None:
        frame_segments.append((start, len(voiced)))

    sample_segments = []
    for s, e in frame_segments:
        start_sample = s * HOP_LENGTH
        end_sample = min(len(y), e * HOP_LENGTH)
        duration = (end_sample - start_sample) / sr
        sample_segments.append((start_sample, end_sample, duration))

    return sample_segments


def shorten_audio_segment(seg: np.ndarray, keep_ratio: float) -> np.ndarray:
    if len(seg) < MIN_SEGMENT_SAMPLES:
        return seg

    # librosa time_stretch: rate > 1 speeds up (shortens)
    rate = 1.0 / keep_ratio
    try:
        return librosa.effects.time_stretch(seg, rate=rate)
    except Exception:
        return seg


def process_file(input_path: str, output_path: str):
    y = load_audio(input_path, TARGET_SR)
    segments = find_voiced_segments(y, TARGET_SR)

    pieces = []
    last = 0
    modified_count = 0

    for start, end, duration in segments:
        if start > last:
            pieces.append(y[last:start])

        voiced_seg = y[start:end]

        if duration >= MIN_ELONGATED_SEC:
            voiced_seg = shorten_audio_segment(voiced_seg, SHORTEN_FACTOR)
            modified_count += 1

        pieces.append(voiced_seg)
        last = end

    if last < len(y):
        pieces.append(y[last:])

    y_out = np.concatenate(pieces) if pieces else y
    sf.write(output_path, y_out, TARGET_SR)

    return {
        "input_path": input_path,
        "output_path": output_path,
        "original_duration_sec": len(y) / TARGET_SR,
        "new_duration_sec": len(y_out) / TARGET_SR,
        "modified_segments": modified_count,
    }


def main():
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    rows_out = []
    summary = []

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        input_audio = row["audio"].strip()
        text = row["text"].strip()

        out_name = Path(input_audio).name
        output_audio = os.path.join(OUTPUT_AUDIO_DIR, out_name)

        info = process_file(input_audio, output_audio)

        rows_out.append({
            "audio": output_audio,
            "text": text,
        })
        summary.append(info)

        print(
            f"Processed {out_name}: "
            f"{info['original_duration_sec']:.2f}s -> {info['new_duration_sec']:.2f}s, "
            f"modified_segments={info['modified_segments']}"
        )

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "text"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nSaved shortened CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()