import os
import re
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_filename_metadata(filename):
    metadata = {
        "tp": "NA",
        "bs": "NA",
        "conc": "NA",
        "prefill": "NA",
    }

    patterns = {
        "tp": r"tp(\d+)",
        "bs": r"bs(\d+)",
        "conc": r"conc(\d+)",
        "prefill": r"prefill(\d+[kK]?)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, filename)
        if match:
            metadata[key] = match.group(1)

    return metadata


def extract_metrics(log_dir):
    metric_patterns = {
        "Total Token throughput (tok/s)": r"Total Token throughput \(tok/s\):\s+([\d.]+)",
        "Median E2EL (ms)": r"Median E2EL \(ms\):\s+([\d.]+)",
        "Median TTFT (ms)": r"Median TTFT \(ms\):\s+([\d.]+)",
        "Median ITL (ms)": r"Median ITL \(ms\):\s+([\d.]+)",
    }

    block_pattern = re.compile(
        r"============ Serving Benchmark Result ============.*?==================================================",
        re.DOTALL,
    )

    rows = []

    for filepath in glob.glob(os.path.join(log_dir, "*")):
        if not os.path.isfile(filepath):
            continue

        filename = os.path.basename(filepath)
        metadata = parse_filename_metadata(filename)

        with open(filepath, "r") as f:
            content = f.read()

        blocks = block_pattern.findall(content)

        for idx, block in enumerate(blocks):
            row = {
                "filename": filename,
                "block_id": idx + 1,
                **metadata,
            }

            for key, pattern in metric_patterns.items():
                match = re.search(pattern, block)
                row[key] = float(match.group(1)) if match else None

            rows.append(row)

    return pd.DataFrame(rows)


def generate_summary(df):
    numeric_cols = [
        "Total Token throughput (tok/s)",
        "Median E2EL (ms)",
        "Median TTFT (ms)",
        "Median ITL (ms)",
    ]

    summary = (
        df.groupby("filename")[numeric_cols]
        .mean()
        .reset_index()
    )

    return summary


def generate_best_config(summary_df):
    if summary_df.empty:
        return pd.DataFrame()

    return summary_df.sort_values(
        by="Total Token throughput (tok/s)",
        ascending=False,
    ).head(1)


def plot_throughput(summary_df):
    if summary_df.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(summary_df.index, summary_df["Total Token throughput (tok/s)"])
    plt.xlabel("Config Index")
    plt.ylabel("Throughput (tok/s)")
    plt.title("Throughput Comparison")
    plt.tight_layout()
    plt.savefig("throughput_plot.png")
    plt.close()


def plot_latency(summary_df):
    if summary_df.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(summary_df.index, summary_df["Median E2EL (ms)"])
    plt.xlabel("Config Index")
    plt.ylabel("Median E2EL (ms)")
    plt.title("Latency Comparison")
    plt.tight_layout()
    plt.savefig("latency_plot.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    args = parser.parse_args()

    df = extract_metrics(args.log_dir)

    raw_output = "raw_result.csv"
    summary_output = "summary_result.csv"
    best_output = "best_config.csv"

    df.to_csv(raw_output, index=False)

    summary_df = generate_summary(df)
    summary_df.to_csv(summary_output, index=False)

    best_df = generate_best_config(summary_df)
    best_df.to_csv(best_output, index=False)

    plot_throughput(summary_df)
    plot_latency(summary_df)

    print("\n=== Raw Result ===")
    print(df)

    print("\n=== Summary Result ===")
    print(summary_df)

    print("\n=== Best Config ===")
    print(best_df)
