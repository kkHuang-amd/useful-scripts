#!/usr/bin/env python3
"""Compare per-device JSON tensor-contract dumps from two runtimes."""

import argparse
import json
from pathlib import Path


DEFAULT_SCALARS = (
    "num_tokens",
    "quant_type",
    "activation",
    "gate_mode",
    "beta",
    "linear_beta",
    "hidden_pad",
    "intermediate_pad",
    "doweight_stage1",
    "is_stream_capturing",
)
DEFAULT_TENSOR_FIELDS = (
    "shape",
    "stride",
    "dtype",
    "contiguous",
    "alignment_16",
    "alignment_64",
    "alignment_256",
    "sample_sha256",
    "is_shuffled",
)


def normalize(value):
    if isinstance(value, str) and "." in value:
        return value.rsplit(".", 1)[-1].lower()
    return value


def load_by_device(directory, device_tensor):
    result = {}
    for path in directory.glob("*.json"):
        data = json.loads(path.read_text())
        tensor = data.get(device_tensor)
        if not tensor or "device" not in tensor:
            continue
        device = int(tensor["device"].rsplit(":", 1)[-1])
        result[device] = {"path": str(path), "data": data}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--device-tensor", default="hidden_states")
    parser.add_argument("--tensors", nargs="+", required=True)
    parser.add_argument("--scalars", nargs="*", default=DEFAULT_SCALARS)
    parser.add_argument("--tensor-fields", nargs="*", default=DEFAULT_TENSOR_FIELDS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    left = load_by_device(args.left_dir, args.device_tensor)
    right = load_by_device(args.right_dir, args.device_tensor)
    comparisons = {}
    for device in sorted(set(left) & set(right)):
        left_data = left[device]["data"]
        right_data = right[device]["data"]
        scalar_diff = {
            field: {
                args.left_name: left_data.get(field),
                args.right_name: right_data.get(field),
            }
            for field in args.scalars
            if normalize(left_data.get(field)) != normalize(right_data.get(field))
        }
        tensor_diff = {}
        for tensor_name in args.tensors:
            left_tensor = left_data.get(tensor_name)
            right_tensor = right_data.get(tensor_name)
            if left_tensor is None or right_tensor is None:
                if left_tensor != right_tensor:
                    tensor_diff[tensor_name] = {
                        args.left_name: left_tensor,
                        args.right_name: right_tensor,
                    }
                continue
            field_diff = {
                field: {
                    args.left_name: left_tensor.get(field),
                    args.right_name: right_tensor.get(field),
                }
                for field in args.tensor_fields
                if left_tensor.get(field) != right_tensor.get(field)
            }
            if field_diff:
                tensor_diff[tensor_name] = field_diff
        comparisons[str(device)] = {
            f"{args.left_name}_path": left[device]["path"],
            f"{args.right_name}_path": right[device]["path"],
            "scalar_diff": scalar_diff,
            "tensor_diff": tensor_diff,
        }

    output = {
        "matched_devices": sorted(int(device) for device in comparisons),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        f"matched_devices={len(comparisons)} "
        f"devices={','.join(comparisons) or 'none'}"
    )


if __name__ == "__main__":
    main()
