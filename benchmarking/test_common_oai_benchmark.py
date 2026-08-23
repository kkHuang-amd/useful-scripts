import asyncio
import unittest

import aiohttp
from aiohttp import web

from common_oai_benchmark import (
    FirstTokenProfiler,
    RequestResult,
    request_one,
    summarize,
)


class TestCommonBenchmark(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.profile_calls = []
        self.stop_event = asyncio.Event()

        async def completions(request):
            body = await request.json()
            response = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"}
            )
            await response.prepare(request)
            await response.write(
                b'data: {"choices":[{"text":"a"}],"usage":null}\n\n'
            )
            await response.write(
                (
                    'data: {"choices":[{"text":"b"}],"usage":'
                    f'{{"prompt_tokens":3,"completion_tokens":{body["max_tokens"]}}}}}\n\n'
                ).encode()
            )
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response

        async def blocking_completions(request):
            body = await request.json()
            response = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"}
            )
            await response.prepare(request)
            await response.write(
                b'data: {"choices":[{"text":"first"}],"usage":null}\n\n'
            )
            await self.stop_event.wait()
            await response.write(
                (
                    'data: {"choices":[{"text":"last"}],"usage":'
                    f'{{"prompt_tokens":3,"completion_tokens":{body["max_tokens"]}}}}}\n\n'
                ).encode()
            )
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response

        async def failed_completions(request):
            return web.Response(status=500, text="measured request failed")

        async def start_profile(request):
            self.profile_calls.append(("start", await request.json()))
            return web.json_response({"started": True})

        async def stop_profile(request):
            self.profile_calls.append(("stop", await request.json()))
            self.stop_event.set()
            return web.json_response({"stopped": True})

        app = web.Application()
        app.router.add_post("/v1/completions", completions)
        app.router.add_post("/blocking/completions", blocking_completions)
        app.router.add_post("/failed/completions", failed_completions)
        app.router.add_post("/start_profile", start_profile)
        app.router.add_post("/stop_profile", stop_profile)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        port = self.site._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self.url = self.base_url + "/v1/completions"

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def test_stream_lifecycle(self):
        async with aiohttp.ClientSession() as session:
            result = await request_one(
                session,
                __import__("asyncio").Semaphore(1),
                self.url,
                "model",
                "abc",
                3,
                2,
                0,
            )
        self.assertTrue(result.success, result.error)
        self.assertTrue(result.done_seen)
        self.assertEqual(result.completion_tokens, 2)
        self.assertEqual(result.text_chunks, 2)

    async def test_concurrent_first_tokens_profile_once_without_deadlock(self):
        profiler = FirstTokenProfiler(
            "sglang",
            self.base_url,
            "/tmp/traces",
            0.01,
            1,
            after_first_tokens=2,
        )
        connector = aiohttp.TCPConnector(limit=2, limit_per_host=2)
        semaphore = asyncio.Semaphore(2)
        async with aiohttp.ClientSession(connector=connector) as session:
            results = await asyncio.wait_for(
                asyncio.gather(
                    *[
                        request_one(
                            session,
                            semaphore,
                            self.base_url + "/blocking/completions",
                            "model",
                            "abc",
                            3,
                            2,
                            request_id,
                            profiler.trigger,
                        )
                        for request_id in range(2)
                    ]
                ),
                timeout=1,
            )
        await profiler.finish()

        self.assertTrue(all(result.success for result in results))
        self.assertTrue(profiler.result["success"], profiler.result["error"])
        self.assertEqual([name for name, _ in self.profile_calls], ["start", "stop"])
        self.assertEqual(profiler.result["requested_threshold"], 2)
        self.assertEqual(profiler.result["observed_count"], 2)
        self.assertEqual(
            {item["request_id"] for item in profiler.result["first_token_observations"]},
            {0, 1},
        )
        self.assertIn(profiler.result["trigger_request_id"], {0, 1})
        self.assertEqual(
            self.profile_calls[0][1],
            {
                "output_dir": "/tmp/traces",
                "activities": ["CPU", "GPU"],
                "with_stack": False,
                "record_shapes": False,
                "profile_by_stage": False,
                "merge_profiles": False,
            },
        )
        self.assertEqual(self.profile_calls[1][1], {})

    async def test_atom_uses_empty_profile_payloads(self):
        profiler = FirstTokenProfiler("atom", self.base_url, None, 0.001, 1)
        profiler.trigger(7, 123)
        await profiler.finish()

        self.assertTrue(profiler.result["success"], profiler.result["error"])
        self.assertEqual(self.profile_calls, [("start", {}), ("stop", {})])

    async def test_timed_profile_before_wave_still_stops_on_timer(self):
        profiler = FirstTokenProfiler(
            "sglang",
            self.base_url,
            "/tmp/traces",
            0.01,
            1,
            mode="before_wave",
        )
        await profiler.start_before_wave()
        self.assertEqual([name for name, _ in self.profile_calls], ["start"])

        async with aiohttp.ClientSession() as session:
            result = await asyncio.wait_for(
                request_one(
                    session,
                    asyncio.Semaphore(1),
                    self.base_url + "/blocking/completions",
                    "model",
                    "abc",
                    3,
                    2,
                    0,
                    profiler.trigger,
                ),
                timeout=1,
            )
        await profiler.finish()

        self.assertTrue(result.success, result.error)
        self.assertTrue(profiler.result["success"], profiler.result["error"])
        self.assertEqual([name for name, _ in self.profile_calls], ["start", "stop"])
        self.assertLess(
            profiler.result["stop_request_perf_counter_ns"], result.done_ns
        )
        self.assertEqual(profiler.result["mode"], "timed")
        self.assertFalse(profiler.summary["config"]["profile_on_first_token"])
        self.assertTrue(profiler.summary["config"]["profile_before_wave"])

    async def test_wave_profile_starts_before_and_stops_after_final_request(self):
        profiler = FirstTokenProfiler(
            "sglang",
            self.base_url,
            "/tmp/traces",
            0.001,
            1,
            mode="before_wave",
            stop_after_wave=True,
        )
        await profiler.start_before_wave()
        self.assertEqual([name for name, _ in self.profile_calls], ["start"])

        connector = aiohttp.TCPConnector(limit=2, limit_per_host=2)
        semaphore = asyncio.Semaphore(2)
        async with aiohttp.ClientSession(connector=connector) as session:
            results = await asyncio.gather(
                *[
                    request_one(
                        session,
                        semaphore,
                        self.url,
                        "model",
                        "abc",
                        3,
                        2,
                        request_id,
                        profiler.trigger,
                    )
                    for request_id in range(2)
                ]
            )
        self.assertEqual([name for name, _ in self.profile_calls], ["start"])
        profiler.mark_wave_complete()
        profiler.signal_wave_complete()
        await asyncio.wait_for(profiler.finish(), timeout=1)

        self.assertTrue(all(result.success for result in results))
        self.assertTrue(profiler.result["success"], profiler.result["error"])
        self.assertEqual([name for name, _ in self.profile_calls], ["start", "stop"])
        self.assertLess(
            profiler.result["start_response_perf_counter_ns"],
            min(result.launch_ns for result in results),
        )
        self.assertGreater(
            profiler.result["stop_request_perf_counter_ns"],
            max(result.done_ns for result in results),
        )
        self.assertEqual(profiler.result["mode"], "wave")
        self.assertTrue(profiler.summary["config"]["profile_stop_after_wave"])
        self.assertIn("wave_completed_at", profiler.result)
        self.assertIn("profile_duration_s", profiler.result)

    async def test_wave_profile_stops_after_failed_request(self):
        profiler = FirstTokenProfiler(
            "atom",
            self.base_url,
            None,
            0.001,
            1,
            mode="before_wave",
            stop_after_wave=True,
        )
        await profiler.start_before_wave()
        async with aiohttp.ClientSession() as session:
            result = await request_one(
                session,
                asyncio.Semaphore(1),
                self.base_url + "/failed/completions",
                "model",
                "abc",
                3,
                2,
                0,
                profiler.trigger,
            )
        profiler.mark_wave_complete()
        profiler.signal_wave_complete()
        await profiler.finish()

        self.assertFalse(result.success)
        self.assertIn("HTTP 500", result.error)
        self.assertTrue(profiler.result["success"], profiler.result["error"])
        self.assertEqual(self.profile_calls, [("start", {}), ("stop", {})])

    async def test_duplicate_request_ids_count_once(self):
        profiler = FirstTokenProfiler(
            "atom",
            self.base_url,
            None,
            0.001,
            1,
            after_first_tokens=2,
        )
        profiler.trigger(3, 100)
        profiler.trigger(3, 200)
        self.assertIsNone(profiler.task)
        profiler.trigger(9, 300)
        await profiler.finish()

        self.assertTrue(profiler.result["success"], profiler.result["error"])
        self.assertEqual(profiler.result["observed_count"], 2)
        self.assertEqual(
            [
                (item["request_id"], item["first_token_perf_counter_ns"])
                for item in profiler.result["first_token_observations"]
            ],
            [(3, 100), (9, 300)],
        )
        self.assertEqual(profiler.result["trigger_request_id"], 9)
        self.assertEqual(self.profile_calls, [("start", {}), ("stop", {})])

    async def test_threshold_unmet_does_not_start_profile(self):
        profiler = FirstTokenProfiler(
            "sglang",
            self.base_url,
            "/tmp/traces",
            0.001,
            1,
            after_first_tokens=2,
        )
        profiler.trigger(4, 123)
        await profiler.finish()

        self.assertFalse(profiler.result["triggered"])
        self.assertFalse(profiler.result["success"])
        self.assertEqual(profiler.result["observed_count"], 1)
        self.assertIn("observed 1", profiler.result["error"])
        self.assertIn("required 2", profiler.result["error"])
        self.assertEqual(self.profile_calls, [])

    async def test_warmup_request_does_not_trigger_profiler(self):
        profiler = FirstTokenProfiler(
            "sglang", self.base_url, "/tmp/traces", 0.001, 1
        )
        async with aiohttp.ClientSession() as session:
            result = await request_one(
                session,
                asyncio.Semaphore(1),
                self.url,
                "model",
                "abc",
                3,
                2,
                -1,
            )
        await profiler.finish()

        self.assertTrue(result.success, result.error)
        self.assertFalse(profiler.result["triggered"])
        self.assertEqual(profiler.result["observed_count"], 0)
        self.assertIn("observed 0", profiler.result["error"])
        self.assertEqual(self.profile_calls, [])

    def test_metrics(self):
        result = RequestResult(
            request_id=0,
            prompt_tokens=3,
            completion_tokens=2,
            launch_ns=0,
            first_token_ns=1_000_000_000,
            last_token_ns=3_000_000_000,
            done_ns=3_100_000_000,
            success=True,
        )
        metrics = summarize([result], 4.0)
        self.assertEqual(metrics["total_token_throughput_tok_s"], 1.25)
        self.assertEqual(metrics["mean_tpot_ms"], 2000.0)


if __name__ == "__main__":
    unittest.main()
