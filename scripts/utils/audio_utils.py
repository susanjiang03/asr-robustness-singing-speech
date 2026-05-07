"""
Audio utilities for ASR robustness project
"""
import os
import wave
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


def load_audio(path: str, target_sr: int = 16000):
    """Load audio file and convert to mono 16kHz"""
    audio, sr = sf.read(path)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio


def load_audio_torch(path: str, target_sr: int = 16000):
    """Load audio file and return as torch tensor"""
    audio = load_audio(path, target_sr)
    return torch.tensor(audio, dtype=torch.float32)


def check_audio_file(path: str, min_duration: float = 1.0, max_duration: float = 45.0):
    """Check if audio file meets requirements"""
    if not os.path.exists(path):
        return False, "file not found"

    try:
        audio, sr = sf.read(path)

        # Check if mono
        if len(audio.shape) > 1:
            if audio.shape[1] != 1:
                return False, f"not mono, channels={audio.shape[1]}"

        # Check sample rate
        if sr != 16000:
            return False, f"sample rate {sr} (should be 16000)"

        duration = len(audio) / sr

        if duration < min_duration:
            return False, f"too short ({duration:.2f}s)"

        if duration > max_duration:
            return False, f"too long ({duration:.2f}s)"

        return True, f"ok ({duration:.2f}s)"

    except Exception as e:
        return False, str(e)


def ensure_wav_mono_16k(path: str) -> Path:
    """
    Vosk needs PCM WAV. If the file is already WAV mono 16k, use it directly.
    Otherwise create a temporary converted file next to it.
    """
    input_path = Path(path)

    try:
        with wave.open(str(input_path), "rb") as wf:
            channels = wf.getnchannels()
            sr = wf.getframerate()
            sampwidth = wf.getsampwidth()

        if channels == 1 and sr == 16000 and sampwidth in (2, 4):
            return input_path
    except Exception:
        pass

    temp_path = input_path.with_name(input_path.stem + "_vosk_temp.wav")

    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    sf.write(temp_path, audio, 16000, subtype="PCM_16")
    return temp_path
