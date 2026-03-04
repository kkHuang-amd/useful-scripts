#!/usr/bin/env python3
# launch.py

import argparse
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prefill-attention-backend", dest="prefill", default="triton")
parser.add_argument("-d", "--decode-attention-backend", dest="decode", default="triton")
args = parser.parse_args()

env = os.environ.copy()
env["SGLANG_USE_AITER"] = "1"

cmd = [
    "python3", "-m", "sglang.launch_server",
    "--model-path", "/dockerx/data/models/openai/gpt-oss-120b/",
    "--tp", "8",
    "--trust-remote-code",
    "--chunked-prefill-size", "130172",
    "--max-running-requests", "128",
    "--mem-fraction-static", "0.85",
    "--prefill-attention-backend", args.prefill,
    "--decode-attention-backend", args.decode,
    "--disable-radix-cache",
    "--port", "8000",
]

subprocess.run(cmd, env=env)
