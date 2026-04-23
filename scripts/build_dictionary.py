import csv
import sys

def build_dict(input_csv, output_txt):
    vocab = set()

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"].strip()

            # add full sentence
            vocab.add(text)

            # add short phrases (2–5 chars sliding window)
            for i in range(len(text)):
                for j in range(i+2, min(len(text)+1, i+6)):
                    vocab.add(text[i:j])

    with open(output_txt, "w", encoding="utf-8") as f:
        for w in sorted(vocab):
            f.write(w + "\n")

    print(f"Saved {len(vocab)} entries to {output_txt}")


if __name__ == "__main__":
    build_dict(sys.argv[1], sys.argv[2])