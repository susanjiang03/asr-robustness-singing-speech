import csv
import sys

def main(input_csv, output_csv, n_rows):
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rows = rows[:n_rows]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_csv}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))