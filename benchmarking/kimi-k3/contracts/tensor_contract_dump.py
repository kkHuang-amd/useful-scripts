"""Small importable helpers for deterministic tensor-contract dumps."""

import hashlib
import json
from pathlib import Path

import torch


def tensor_metadata(tensor, sample_elements=4096):
    if tensor is None:
        return None
    pointer = tensor.data_ptr()
    sample = tensor.detach().reshape(-1)[:sample_elements].contiguous().cpu()
    sample_bytes = sample.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": tensor.is_contiguous(),
        "alignment_16": pointer % 16,
        "alignment_64": pointer % 64,
        "alignment_256": pointer % 256,
        "sample_elements": sample.numel(),
        "sample_sha256": hashlib.sha256(sample_bytes).hexdigest(),
    }


def dump_contract(path, *, scalars=None, tensors=None):
    payload = dict(scalars or {})
    payload.update(
        {
            name: tensor_metadata(tensor)
            for name, tensor in (tensors or {}).items()
        }
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
