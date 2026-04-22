import os
import json
import csv
import argparse

# 欄位（加上 max_concurrency）
TARGET_FIELDS = [
    "max_concurrency",
    "total_token_throughput",
    "median_ttft_ms",
    "median_tpot_ms",
    "median_itl_ms",
    "median_e2el_ms",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="summary.csv")

    args = parser.parse_args()

    results = []

    for root, _, files in os.walk(args.input_dir):
        for file in files:
            file_path = os.path.join(root, file)

            try:
                with open(file_path, "r") as f:
                    content = f.read().strip()

                    for line in content.splitlines():
                        if not line.strip():
                            continue

                        data = json.loads(line)

                        row = {"file": file}
                        for field in TARGET_FIELDS:
                            row[field] = data.get(field, None)

                        results.append(row)

            except Exception as e:
                print(f"Skip {file_path}: {e}")

    # 🔥 依 max_concurrency 排序（None 會被丟到最後）
    results = sorted(
        results,
        key=lambda x: (x["max_concurrency"] is None, x["max_concurrency"])
    )

    # 寫 CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file"] + TARGET_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"Done! Output -> {args.output}")


if __name__ == "__main__":
    main()
