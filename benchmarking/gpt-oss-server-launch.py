#!/usr/bin/env python3
# launch.py

import argparse
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prefill-attention-backend", dest="prefill", default="triton")
parser.add_argument("-d", "--decode-attention-backend", dest="decode", default="triton")
parser.add_argument("--model", dest="model", default="/dockerx/raid/models/gpt-oss-120b/")
parser.add_argument("--fp8-kv", dest="fp8_kv", action="store_true", help="Enable fp8 kv")

# MTP / speculative decoding switch
parser.add_argument("--mtp", action="store_true", help="Enable MTP speculative decoding")
# draft model path (required only when mtp enabled)
parser.add_argument("--draft-model-path", dest="draft_model_path", default=None)

args = parser.parse_args()

env = os.environ.copy()
env["SGLANG_USE_AITER"] = "1"


#"--enable-aiter-allreduce-fusion",

cmd = [
    "python3", "-m", "sglang.launch_server",
    "--model-path", args.model,
    "--tp", "8",
    "--trust-remote-code",
    "--chunked-prefill-size", "131072",
    "--max-running-requests", "128",
    "--mem-fraction-static", "0.85",
    "--prefill-attention-backend", args.prefill,
    "--decode-attention-backend", args.decode,
    "--page-size", "64",
    "--disable-radix-cache",
    "--port", "8000",
]

if args.fp8_kv:
    cmd.extend([
        "--kv-cache-dtype", "fp8_e4m3",
    ])

# Optional MTP speculative decoding
if args.mtp:
    if args.draft_model_path is None:
        raise ValueError("--draft-model-path is required when --mtp is enabled")

    cmd.extend([
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", args.draft_model_path,
        "--speculative-num-steps", "3",
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "4",
    ])

subprocess.run(cmd, env=env)
