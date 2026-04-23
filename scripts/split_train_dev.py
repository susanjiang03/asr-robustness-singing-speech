from __future__ import annotations

import csv
import os
import random
import shutil
import sys
from pathlib import Path


def split_dataset(
    input_csv: str,
    train_csv: str,
    dev_csv: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    copy_files: bool = False,
    train_dir: str | None = None,
    dev_dir: str | None = None,
):
    random.seed(seed)

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from {input_csv}")

    # shuffle
    random.shuffle(rows)

    # split
    split_idx = int(len(rows) * train_ratio)
    train_rows = rows[:split_idx]
    dev_rows = rows[split_idx:]

    print(f"Train: {len(train_rows)}")
    print(f"Dev  : {len(dev_rows)}")

    # optionally copy files
    if copy_files:
        if not train_dir or not dev_dir:
            raise ValueError("train_dir and dev_dir must be provided when copy_files=True")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(dev_dir, exist_ok=True)

        def copy_and_update(rows_subset, target_dir):
            new_rows = []
            for row in rows_subset:
                src = row["audio"]
                filename = Path(src).name
                dst = os.path.join(target_dir, filename)

                shutil.copy(src, dst)

                new_rows.append({
                    "audio": dst,
                    "text": row["text"],
                })
            return new_rows

        train_rows = copy_and_update(train_rows, train_dir)
        dev_rows = copy_and_update(dev_rows, dev_dir)

    # write train.csv
    with open(train_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "text"])
        writer.writeheader()
        writer.writerows(train_rows)

    # write dev.csv
    with open(dev_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "text"])
        writer.writeheader()
        writer.writerows(dev_rows)

    print("\n=== Done ===")
    print(f"Saved train CSV: {train_csv}")
    print(f"Saved dev CSV  : {dev_csv}")

    # sanity check: no overlap
    train_set = set(r["audio"] for r in train_rows)
    dev_set = set(r["audio"] for r in dev_rows)
    overlap = train_set.intersection(dev_set)

    if overlap:
        print("⚠️ Overlap detected!")
    else:
        print("✅ No overlap between train and dev")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 6):
        print("Usage:")
        print("python scripts/split_train_dev.py <input_csv> <train_csv> <dev_csv>")
        print("Optional copy mode:")
        print("python scripts/split_train_dev.py <input_csv> <train_csv> <dev_csv> <train_dir> <dev_dir>")
        sys.exit(1)

    input_csv = sys.argv[1]
    train_csv = sys.argv[2]
    dev_csv = sys.argv[3]

    if len(sys.argv) == 6:
        train_dir = sys.argv[4]
        dev_dir = sys.argv[5]

        split_dataset(
            input_csv,
            train_csv,
            dev_csv,
            copy_files=True,
            train_dir=train_dir,
            dev_dir=dev_dir,
        )
    else:
        split_dataset(
            input_csv,
            train_csv,
            dev_csv,
        )