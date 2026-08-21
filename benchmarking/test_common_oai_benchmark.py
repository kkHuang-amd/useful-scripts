import unittest

import aiohttp
from aiohttp import web

from common_oai_benchmark import RequestResult, request_one, summarize


class TestCommonBenchmark(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
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

        app = web.Application()
        app.router.add_post("/v1/completions", completions)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        port = self.site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/v1/completions"

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
