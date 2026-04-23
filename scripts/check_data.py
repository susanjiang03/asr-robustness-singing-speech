import csv
import os
import soundfile as sf

def check_audio(path):
    if not os.path.exists(path):
        return False, "file not found"

    try:
        audio, sr = sf.read(path)

        # audio shape:
        # mono -> (samples,)
        # stereo -> (samples, channels)
        if len(audio.shape) > 1:
            if audio.shape[1] != 1:
                return False, f"not mono, channels={audio.shape[1]}"

        if sr != 16000:
            return False, f"sample rate {sr} (should be 16000)"

        duration = len(audio) / sr

        if duration < 1:
            return False, f"too short ({duration:.2f}s)"

        # if duration > 20:
        if duration > 45:
            return False, f"too long ({duration:.2f}s)"

        return True, f"ok ({duration:.2f}s)"

    except Exception as e:
        return False, str(e)


def check_csv(csv_path):
    print(f"\nChecking {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        total = 0
        bad = 0

        for row in reader:
            total += 1
            audio = row["audio"]
            ok, msg = check_audio(audio)

            if not ok:
                print(f"[BAD] {audio} → {msg}")
                bad += 1

        print(f"\nTotal: {total}")
        print(f"Bad: {bad}")
        print(f"Good: {total - bad}")


if __name__ == "__main__":
    check_csv("data/test.csv")