import os
import re
import glob
import argparse
import pandas as pd


def extract_metrics(log_dir):
    patterns = {
        "Total token throughput (tok/s)": r"Total token throughput \(tok/s\):\s+([\d.]+)",
        "Median E2E Latency (ms)": r"Median E2E Latency \(ms\):\s+([\d.]+)",
        "Median TTFT (ms)": r"Median TTFT \(ms\):\s+([\d.]+)",
        "Median ITL (ms)": r"Median ITL \(ms\):\s+([\d.]+)",
    }

    rows = []

    for filepath in glob.glob(os.path.join(log_dir, "*")):
        if not os.path.isfile(filepath):
            continue

        with open(filepath, "r") as f:
            content = f.read()

        row = {"filename": os.path.basename(filepath)}

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            row[key] = float(match.group(1)) if match else None

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract benchmark metrics from logs")
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory containing log files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result.csv",
        help="Output CSV file name"
    )

    args = parser.parse_args()

    df = extract_metrics(args.log_dir)
    df.to_csv(args.output, index=False)

    print(df)
    print(f"\nSaved to {args.output}")
