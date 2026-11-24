import json
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import httpx

from mechanisms.proxy import BudgetProxy


class BudgetProxyTests(unittest.TestCase):
    def test_truncated_stream_cost_recovered_before_next_request(self):
        methods = []

        def upstream(request):
            methods.append(request.method)
            if request.method == "GET":
                self.assertEqual(request.url.params["id"], "gen-truncated")
                return httpx.Response(200, json={"data": {"total_cost": 0.12}})
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content='data: {"id":"gen-truncated","choices":[]}\n\n',
            )

        with tempfile.TemporaryDirectory() as directory:
            with BudgetProxy(
                "key",
                "0.10",
                Path(directory) / "budget.json",
                client=httpx.Client(transport=httpx.MockTransport(upstream)),
            ) as proxy:
                self.assertEqual(self.request(proxy).status_code, 200)
                self.assertEqual(self.request(proxy).status_code, 402)
                self.assertEqual(proxy.budget.spent, Decimal("0.12"))
                self.assertEqual(
                    proxy.budget.requests[0]["cost_source"], "generation_lookup"
                )
                self.assertEqual(methods, ["POST", "GET"])

    def request(self, proxy, token=None):
        return httpx.post(
            f"http://127.0.0.1:{proxy.server.server_port}/v1/chat/completions",
            headers={"Authorization": f"Bearer {token or proxy.token}"},
            json={"model": "test", "stream": True},
            timeout=5,
            trust_env=False,
        )

    def test_serialized_streaming_cap(self):
        calls = []

        def upstream(request):
            self.assertEqual(request.headers["Authorization"], "Bearer real-key")
            calls.append(1)
            time.sleep(0.03)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content='data: {"id":"gen-1","usage":{"cost":0.12}}\n\ndata: [DONE]\n\n',
            )

        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "budget.json"
            with BudgetProxy(
                "real-key",
                "0.10",
                ledger,
                client=httpx.Client(transport=httpx.MockTransport(upstream)),
            ) as proxy:
                with ThreadPoolExecutor(2) as pool:
                    responses = list(pool.map(lambda _: self.request(proxy), range(2)))
                self.assertEqual(sorted((r.status_code for r in responses)), [200, 402])
                self.assertEqual(len(calls), 1)
                self.assertEqual(proxy.budget.status, "budget_exhausted")
            recorded = json.loads(ledger.read_text())
            self.assertEqual(recorded["spent_usd"], "0.12")
            self.assertEqual(len(recorded["requests"]), 1)
            self.assertNotIn("real-key", ledger.read_text())

    def test_missing_cost_blocks_next_call(self):
        with tempfile.TemporaryDirectory() as directory:
            client = httpx.Client(
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(200, json={"id": "missing"})
                )
            )
            with BudgetProxy(
                "key", 2, Path(directory) / "budget.json", client=client
            ) as proxy:
                self.assertEqual(self.request(proxy).status_code, 200)
                self.assertEqual(self.request(proxy).status_code, 402)
                self.assertEqual(proxy.budget.status, "cost_unknown")

    def test_model_scope_and_invalid_token(self):
        calls = []

        def upstream(request):
            calls.append(request)
            return httpx.Response(200, json={"choices": [], "usage": {"cost": 0}})

        with tempfile.TemporaryDirectory() as directory:
            with BudgetProxy(
                "upstream-secret",
                1,
                Path(directory) / "budget.json",
                model="allowed",
                client=httpx.Client(transport=httpx.MockTransport(upstream)),
            ) as proxy:
                self.assertEqual(self.request(proxy, token="wrong").status_code, 401)
                self.assertEqual(self.request(proxy).status_code, 403)
                self.assertEqual(calls, [])
