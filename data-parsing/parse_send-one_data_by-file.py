import re
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True, help="log file path")
args = parser.parse_args()

file_path = args.file

rows = []

with open(file_path, "r") as f:
    text = f.read()

pattern = r"\|\s*([\d\.]+)\s*\|\s*(\d+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|"
matches = re.findall(pattern, text)

for m in matches:
    rows.append([
        float(m[0]),
        int(m[1]),
        float(m[2]),
        float(m[3])
    ])

df = pd.DataFrame(rows, columns=[
    "Latency (s)", "Tokens", "Acc Length", "Speed (token/s)"
])

df.loc["Average"] = df.mean(numeric_only=True)

print(df)
