from pydub import AudioSegment
import os

INPUT_DIR = "data/clips/test_m4a"
OUTPUT_DIR = "data/clips/test"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    if file.endswith(".m4a"):
        input_path = os.path.join(INPUT_DIR, file)

        # load m4a
        audio = AudioSegment.from_file(input_path, format="m4a")

        # convert to mono + 16kHz
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)

        # save as wav
        output_file = file.replace(".m4a", ".wav")
        output_path = os.path.join(OUTPUT_DIR, output_file)

        audio.export(output_path, format="wav")

        print(f"Converted: {file} -> {output_file}")

print("Done!")