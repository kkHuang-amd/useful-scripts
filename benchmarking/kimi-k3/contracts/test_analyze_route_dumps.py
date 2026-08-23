import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ANALYZER = Path(__file__).with_name("analyze_route_dumps.py")


def write_dump(root, rank, call, counts, blocks):
    compact = [
        {"expert": expert, "routes": routes}
        for expert, routes in enumerate(counts)
        if routes
    ]
    nonzero = [item["routes"] for item in compact]
    payload = {
        "schema": "k3-route-dump-v1",
        "armed": True,
        "arm_file": str((root / "route.arm").resolve()),
        "arm_timestamp_utc": "2026-08-22T12:00:00+00:00",
        "arm_time_ns": 100,
        "arm_monotonic_ns": 200,
        "dump_timestamp_utc": "2026-08-22T12:00:01+00:00",
        "dump_time_ns": 101 + call,
        "dump_monotonic_ns": 201 + call,
        "rank": rank,
        "call_index": call,
        "total_routes": sum(counts),
        "unique_active_experts": len(nonzero),
        "nonzero_routes": {
            "min": min(nonzero),
            "max": max(nonzero),
            "mean": sum(nonzero) / len(nonzero),
        },
        "expert_count_size": len(counts),
        "expert_counts": compact,
        "full_bincount": counts,
        "bm32_padded_blocks": blocks,
        "bm32_padded_routes": blocks * 32,
        "env_mode": root.name,
        "quant_type": "synthetic",
    }
    (root / f"rank-{rank}-call-{call:03d}.json").write_text(json.dumps(payload))


class AnalyzeRouteDumpsTest(unittest.TestCase):
    def test_exact_alignment_and_deltas(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            left, right, output = base / "left", base / "right", base / "out"
            left.mkdir()
            right.mkdir()
            for rank in (0, 1):
                write_dump(left, rank, 0, [32, 32], 2)
                write_dump(left, rank, 1, [31, 33], 3)
                write_dump(right, rank, 0, [16, 48], 3)
                write_dump(right, rank, 1, [31, 33], 3)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ANALYZER),
                    str(left),
                    str(right),
                    "--left-name",
                    "a8w4",
                    "--right-name",
                    "a16w4",
                    "--output-dir",
                    str(output),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((output / "route-analysis.json").read_text())
            self.assertEqual(payload["schema"], "k3-route-analysis-v2")
            self.assertTrue(payload["arming"]["a8w4"]["armed"])
            self.assertTrue(payload["arming"]["a8w4"]["all_dumps_after_arm"])
            self.assertTrue(payload["alignment"]["exact"])
            self.assertEqual(payload["alignment"]["aligned_count"], 4)
            self.assertEqual(
                payload["cross_engine"]["exact_expert_count_matches"], 2
            )
            first = payload["cross_engine"]["per_layer"][0]
            self.assertEqual(first["bm32_padded_blocks_delta"], 1)
            self.assertEqual(first["expert_count_l1_delta"], 32)
            self.assertTrue((output / "route-layers.csv").is_file())
            self.assertTrue((output / "route-deltas.csv").is_file())
            report = (output / "route-analysis.md").read_text()
            self.assertIn("no valid timing comparison", report)

    def test_incomplete_alignment_fails_after_writing_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            left, right, output = base / "left", base / "right", base / "out"
            left.mkdir()
            right.mkdir()
            write_dump(left, 0, 0, [32, 32], 2)
            write_dump(right, 0, 1, [32, 32], 2)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ANALYZER),
                    str(left),
                    str(right),
                    "--output-dir",
                    str(output),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 1)
            payload = json.loads((output / "route-analysis.json").read_text())
            self.assertFalse(payload["alignment"]["exact"])
            self.assertEqual(len(payload["alignment"]["left_only"]), 1)
            self.assertEqual(len(payload["alignment"]["right_only"]), 1)

    def test_old_unarmed_dump_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            left, right, output = base / "left", base / "right", base / "out"
            left.mkdir()
            right.mkdir()
            write_dump(left, 0, 0, [32, 32], 2)
            write_dump(right, 0, 0, [32, 32], 2)
            path = next(left.glob("*.json"))
            payload = json.loads(path.read_text())
            payload.pop("armed")
            payload.pop("arm_time_ns")
            path.write_text(json.dumps(payload))
            result = subprocess.run(
                [
                    sys.executable,
                    str(ANALYZER),
                    str(left),
                    str(right),
                    "--output-dir",
                    str(output),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("old or unarmed route dump is invalid", result.stderr)


if __name__ == "__main__":
    unittest.main()
