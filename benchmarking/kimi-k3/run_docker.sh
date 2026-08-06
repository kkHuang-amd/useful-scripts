#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-}"
NAME="${NAME:-kimi-k3-bench}"

if [[ -z "$IMAGE" ]]; then
  echo "usage: IMAGE=<latest-sglang-image@digest> $0 [container command ...]" >&2
  exit 2
fi

if [[ "$IMAGE" != *@sha256:* ]]; then
  echo "warning: IMAGE is not digest-pinned; record the resolved digest" >&2
fi

command=("$@")
if [[ "${#command[@]}" -eq 0 ]]; then
  command=(bash)
fi

exec docker run --rm -it \
  --name "$NAME" \
  --network host \
  --ipc host \
  --shm-size 32g \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /dockerx/data/models/Kimi-K3:/dockerx/data/models/Kimi-K3:ro \
  -v /dockerx/data/models/Kimi-K3-DSpark:/dockerx/data/models/Kimi-K3-DSpark:ro \
  -v /sgl-workspace/kvv-bench/kvv-k3-0727-update:/sgl-workspace/kvv-bench/kvv-k3-0727-update \
  -v /dockerx/var/amdsgl/kk/workspace/useful-scripts/benchmarking/kimi-k3:/opt/kimi-k3-scripts:ro \
  "$IMAGE" \
  "${command[@]}"

