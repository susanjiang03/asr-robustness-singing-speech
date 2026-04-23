# import os
# import soundfile as sf
# import numpy as np

# INPUT_DIR = "data/clips/test"
# OUTPUT_DIR = "data/clips/test_mono"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# for file in os.listdir(INPUT_DIR):
#     if file.endswith(".wav"):
#         path = os.path.join(INPUT_DIR, file)

#         audio, sr = sf.read(path)

#         # convert stereo → mono
#         if len(audio.shape) > 1:
#             audio = np.mean(audio, axis=1)

#         out_path = os.path.join(OUTPUT_DIR, file)
#         sf.write(out_path, audio, sr)

#         print(f"Converted: {file}")

# print("Done!")

import os
import soundfile as sf
import numpy as np
import librosa

INPUT_DIR = "data/clips/test"
OUTPUT_DIR = "data/clips/test_fixed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SR = 16000

for file in os.listdir(INPUT_DIR):
    if file.endswith(".wav"):
        path = os.path.join(INPUT_DIR, file)

        audio, sr = sf.read(path)

        # stereo → mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # resample to 16kHz
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

        out_path = os.path.join(OUTPUT_DIR, file)
        sf.write(out_path, audio, TARGET_SR)

        print(f"Fixed: {file}")

print("Done!")