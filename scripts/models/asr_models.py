"""
ASR model wrappers for evaluation
"""
import json
import wave
from pathlib import Path

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
from vosk import Model as VoskModel, KaldiRecognizer

from ..utils.audio_utils import ensure_wav_mono_16k, load_audio_torch
from ..utils.text_utils import normalize_text


class ASRModel:
    """Base class for ASR models"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError


class WhisperASR(ASRModel):
    """Whisper ASR model wrapper"""
    
    def __init__(self, model_name: str = "openai/whisper-small", device: str = "cpu"):
        super().__init__(device)
        self.model_name = model_name
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    
    def transcribe(self, audio_path: str) -> str:
        waveform = load_audio_torch(audio_path)

        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )

        with torch.no_grad():
            pred_ids = self.model.generate(
                inputs["input_features"].to(self.device),
                attention_mask=inputs.get("attention_mask", None),
                language="zh",
                task="transcribe",
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                max_new_tokens=128,
                do_sample=False,
            )

        return self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]


class Wav2Vec2ASR(ASRModel):
    """Wav2Vec2 ASR model wrapper"""
    
    def __init__(self, model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn", device: str = "cpu"):
        super().__init__(device)
        self.model_name = model_name
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    
    def transcribe(self, audio_path: str) -> str:
        waveform = load_audio_torch(audio_path)

        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits

        pred_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(pred_ids)[0]


class VoskASR(ASRModel):
    """Vosk ASR model wrapper"""
    
    def __init__(self, model_path: str):
        super().__init__("cpu")  # Vosk runs on CPU
        self.model_path = model_path
        self.model = VoskModel(model_path)
    
    def transcribe(self, audio_path: str) -> str:
        wav_path = ensure_wav_mono_16k(audio_path)

        wf = wave.open(str(wav_path), "rb")
        rec = KaldiRecognizer(self.model, wf.getframerate())

        result_text = ""

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break

            if rec.AcceptWaveform(data):
                part = json.loads(rec.Result())
                result_text += part.get("text", "")

        final_part = json.loads(rec.FinalResult())
        result_text += final_part.get("text", "")

        wf.close()
        return result_text


def get_model(model_name: str, device: str = "cpu", vosk_model_path: str = None) -> ASRModel:
    """Factory function to get ASR model"""
    if model_name == "whisper":
        return WhisperASR(device=device)
    elif model_name == "wav2vec2":
        return Wav2Vec2ASR(device=device)
    elif model_name == "vosk":
        if not vosk_model_path:
            raise ValueError("vosk_model_path is required for Vosk model")
        return VoskASR(vosk_model_path)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
