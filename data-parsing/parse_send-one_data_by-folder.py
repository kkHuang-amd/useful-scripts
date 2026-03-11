import os
import re
import pandas as pd
import argparse


def parse_log_file(file_path):
    rows = []

    with open(file_path, "r") as f:
        text = f.read()

    pattern = r"\|\s*([\d\.]+)\s*\|\s*(\d+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|"
    matches = re.findall(pattern, text)

    for m in matches:
        rows.append({
            "file": os.path.basename(file_path),
            "Latency (s)": float(m[0]),
            "Tokens": int(m[1]),
            "Acc Length": float(m[2]),
            "Speed (token/s)": float(m[3])
        })

    return rows


def main(log_dir, output):
    all_rows = []

    for file in os.listdir(log_dir):
        if file.endswith(".log") or file.endswith(".txt"):
            path = os.path.join(log_dir, file)
            all_rows.extend(parse_log_file(path))

    df = pd.DataFrame(all_rows)

    if df.empty:
        print("No valid data found.")
        return

    avg_row = df.mean(numeric_only=True)
    avg_row["file"] = "Average"

    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    print(df)

    df.to_csv(output, index=False)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="summary.csv")

    args = parser.parse_args()

    main(args.log_dir, args.output)
