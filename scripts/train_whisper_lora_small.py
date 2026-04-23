from __future__ import annotations

import os
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
# Prepare dataset
# -------------------------
def prepare(batch: dict[str, Any], processor: WhisperProcessor):
    audio = load_audio(batch["audio"])
    text = normalize(batch["text"])

    inputs = processor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="np",
    )

    labels = processor.tokenizer(text).input_ids

    return {
        "input_features": inputs.input_features[0],
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

    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        language=LANGUAGE,
        task=TASK,
    )

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.config.use_cache = False

    # -------------------------
    # LoRA config
    # -------------------------
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------------------------
    # Load dataset
    # -------------------------
    dataset = load_dataset(
        "csv",
        data_files={
            "train": train_csv,
            "validation": dev_csv,
        },
    )

    dataset = dataset.map(
        lambda x: prepare(x, processor),
        remove_columns=dataset["train"].column_names,
    )

    # -------------------------
    # Training args
    # -------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        num_train_epochs=1,

        do_eval=False,
        logging_steps=1,
        save_steps=100000,   # effectively don't save during training
        save_total_limit=1,

        fp16=False,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollator(processor),
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    print(f"\nSaved model to: {final_dir}")


# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage:")
        print("python train_whisper_lora.py <train_csv> <dev_csv> <output_dir>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])