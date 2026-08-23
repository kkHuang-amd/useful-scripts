import importlib.util
import json
import os
import sys
import tempfile
import time
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import torch


SITE_PATH = Path(__file__).with_name("sitecustomize.py")


def load_overlay(name):
    spec = importlib.util.spec_from_file_location(name, SITE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_arm(path):
    state = {
        "schema": "k3-route-arm-v1",
        "armed": True,
        "armed_at_utc": datetime.now(timezone.utc).isoformat(),
        "armed_time_ns": time.time_ns(),
        "armed_monotonic_ns": time.monotonic_ns(),
    }
    path.write_text(json.dumps(state))
    return state


class SiteCustomizeTest(unittest.TestCase):
    def setUp(self):
        self.original_aiter = sys.modules.get("aiter")
        self.original_fused_moe = sys.modules.get("aiter.fused_moe")
        package = types.ModuleType("aiter")
        package.__path__ = []
        fused_module = types.ModuleType("aiter.fused_moe")

        def fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, **kwargs):
            return hidden_states

        fused_module.fused_moe = fused_moe
        package.fused_moe = fused_module
        sys.modules["aiter"] = package
        sys.modules["aiter.fused_moe"] = fused_module
        self.fused_module = fused_module

    def tearDown(self):
        for name, value in (
            ("aiter", self.original_aiter),
            ("aiter.fused_moe", self.original_fused_moe),
        ):
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        for name in list(sys.modules):
            if name.startswith("_test_k3_sitecustomize_"):
                sys.modules.pop(name)

    def test_noop_without_environment(self):
        original = self.fused_module.fused_moe
        with mock.patch.dict(os.environ, {}, clear=True):
            load_overlay("_test_k3_sitecustomize_noop")
        self.assertIs(self.fused_module.fused_moe, original)

    def test_monkeypatch_idempotence_counts_padding_and_return(self):
        with tempfile.TemporaryDirectory() as temporary:
            arm_file = Path(temporary) / "route.arm"
            environment = {
                "K3_ROUTE_DUMP_DIR": temporary,
                "K3_ROUTE_DUMP_ARM_FILE": str(arm_file),
                "K3_ROUTE_DUMP_MAX_CALLS": "92",
                "K3_ROUTE_ENV_MODE": "synthetic-cpu",
                "RANK": "3",
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                load_overlay("_test_k3_sitecustomize_first")
                wrapped = self.fused_module.fused_moe
                load_overlay("_test_k3_sitecustomize_second")
                self.assertIs(self.fused_module.fused_moe, wrapped)

                hidden = torch.randn(64, 8)
                topk_ids = (torch.arange(1024) % 3).reshape(64, 16)
                # A qualifying startup call before arming must neither dump nor
                # consume call index zero.
                wrapped(
                    hidden,
                    torch.empty(3, 1, 1),
                    torch.empty(3, 1, 1),
                    torch.ones(64, 16),
                    topk_ids,
                )
                self.assertFalse(list(Path(temporary).glob("*.json")))
                arm_state = write_arm(arm_file)
                result = wrapped(
                    hidden,
                    torch.empty(3, 1, 1),
                    torch.empty(3, 1, 1),
                    torch.ones(64, 16),
                    topk_ids,
                )

            self.assertIs(result, hidden)
            paths = list(Path(temporary).glob("*.json"))
            self.assertEqual(len(paths), 1)
            payload = json.loads(paths[0].read_text())
            self.assertEqual(payload["call_index"], 0)
            self.assertTrue(payload["armed"])
            self.assertEqual(payload["arm_file"], str(arm_file.resolve()))
            self.assertEqual(payload["arm_time_ns"], arm_state["armed_time_ns"])
            self.assertGreaterEqual(
                payload["dump_time_ns"], payload["arm_time_ns"]
            )
            self.assertGreaterEqual(
                payload["dump_monotonic_ns"], payload["arm_monotonic_ns"]
            )
            self.assertTrue(payload["dump_timestamp_utc"])
            self.assertEqual(payload["rank"], 3)
            self.assertEqual(payload["total_routes"], 1024)
            self.assertEqual(payload["unique_active_experts"], 3)
            self.assertEqual(payload["nonzero_routes"]["min"], 341)
            self.assertEqual(payload["nonzero_routes"]["max"], 342)
            self.assertAlmostEqual(payload["nonzero_routes"]["mean"], 1024 / 3)
            self.assertEqual(payload["full_bincount"], [342, 341, 341])
            self.assertEqual(payload["bm32_padded_blocks"], 33)
            self.assertFalse(list(Path(temporary).glob("*.tmp")))

    def test_shape_and_capture_filters(self):
        with tempfile.TemporaryDirectory() as temporary:
            arm_file = Path(temporary) / "route.arm"
            write_arm(arm_file)
            with mock.patch.dict(
                os.environ,
                {
                    "K3_ROUTE_DUMP_DIR": temporary,
                    "K3_ROUTE_DUMP_ARM_FILE": str(arm_file),
                },
                clear=True,
            ):
                overlay = load_overlay("_test_k3_sitecustomize_filters")
                wrapped = self.fused_module.fused_moe
                args = (
                    torch.randn(64, 8),
                    torch.empty(3, 1, 1),
                    torch.empty(3, 1, 1),
                    torch.ones(64, 16),
                )
                wrapped(*args, torch.zeros(63, 16, dtype=torch.int64))
                with mock.patch.object(overlay, "_is_stream_capturing", return_value=True):
                    wrapped(*args, torch.zeros(64, 16, dtype=torch.int64))
            dumps = [
                path
                for path in Path(temporary).glob("*.json")
                if path != arm_file
            ]
            self.assertFalse(dumps)


if __name__ == "__main__":
    unittest.main()
