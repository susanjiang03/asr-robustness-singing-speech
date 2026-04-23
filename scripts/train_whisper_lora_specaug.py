from __future__ import annotations

import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
from datasets import load_dataset
from opencc import OpenCC
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

MODEL_NAME = "openai/whisper-small"
LANGUAGE = "zh"
TASK = "transcribe"
TARGET_SR = 16000

cc = OpenCC("t2s")


# -------------------------
# Text normalization
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
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio.astype(np.float32)


# -------------------------
# SpecAugment on mel features
# -------------------------
def apply_specaugment(
    features: np.ndarray,
    time_masks: int = 2,
    freq_masks: int = 2,
    max_time_width: int = 12,
    max_freq_width: int = 8,
) -> np.ndarray:
    """
    features shape: (n_mels, time_steps)
    """
    aug = features.copy()
    n_mels, t = aug.shape

    # frequency masking
    for _ in range(freq_masks):
        if n_mels <= 1:
            break
        width = random.randint(0, min(max_freq_width, n_mels - 1))
        if width == 0:
            continue
        f0 = random.randint(0, n_mels - width)
        aug[f0:f0 + width, :] = 0.0

    # time masking
    for _ in range(time_masks):
        if t <= 1:
            break
        width = random.randint(0, min(max_time_width, t - 1))
        if width == 0:
            continue
        t0 = random.randint(0, t - width)
        aug[:, t0:t0 + width] = 0.0

    return aug


# -------------------------
# Prepare dataset
# -------------------------
def prepare(batch: dict[str, Any], processor: WhisperProcessor, use_specaugment: bool):
    audio = load_audio(batch["audio"])
    text = normalize(batch["text"])

    inputs = processor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="np",
    )

    input_features = inputs.input_features[0]  # shape (80, T)

    if use_specaugment:
        input_features = apply_specaugment(input_features)

    labels = processor.tokenizer(text).input_ids

    return {
        "input_features": input_features,
        "labels": labels,
    }


# -------------------------
# Data collator
# -------------------------
@dataclass
class DataCollator:
    processor: WhisperProcessor

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch


# -------------------------
# Main
# -------------------------
def main(train_csv: str, dev_csv: str, output_dir: str):
    print("Train:", train_csv)
    print("Dev  :", dev_csv)
    print("Out  :", output_dir)

    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        language=LANGUAGE,
        task=TASK,
    )

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "csv",
        data_files={
            "train": train_csv,
            "validation": dev_csv,
        },
    )

    train_dataset = dataset["train"].map(
        lambda x: prepare(x, processor, use_specaugment=True),
        remove_columns=dataset["train"].column_names,
    )

    val_dataset = dataset["validation"].map(
        lambda x: prepare(x, processor, use_specaugment=False),
        remove_columns=dataset["validation"].column_names,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        num_train_epochs=1,
        do_eval=False,
        logging_steps=1,
        save_steps=100000,
        save_total_limit=1,
        fp16=False,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollator(processor),
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    print(f"\nSaved model to: {final_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage:")
        print("python scripts/train_whisper_lora_specaug.py <train_csv> <dev_csv> <output_dir>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])