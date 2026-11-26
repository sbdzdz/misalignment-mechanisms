import json
import socket
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import httpx

from mechanisms.broker import broker_request
from mechanisms.proxy import BudgetProxy


class BrokerRequestTests(unittest.TestCase):
    def test_missing_reply_is_a_runtime_error(self):
        """A broker that hangs up without replying raises RuntimeError."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broker.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(path))
                server.listen(1)

                def hang_up():
                    """Accept one request and close without replying."""
                    connection, _ = server.accept()
                    connection.recv(65536)
                    connection.close()

                thread = threading.Thread(target=hang_up)
                thread.start()
                with self.assertRaises(RuntimeError):
                    broker_request(
                        {"action": "close", "run_id": "x"},
                        {"socket": str(path), "request_timeout_sec": 1},
                    )
                thread.join()


class BudgetProxyTests(unittest.TestCase):
    def test_close_does_not_wait_for_in_flight_request(self):
        """Closing returns at once; the late reply still records its cost."""
        release = threading.Event()

        def upstream(request):
            """Answer only after the test releases the reply.

            Args:
                request: Forwarded model request.

            Returns:
                Response with a known cost.
            """
            release.wait(5)
            return httpx.Response(200, json={"id": "gen-late", "usage": {"cost": 0.01}})

        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "budget.json"
            proxy = BudgetProxy(
                "key",
                1,
                ledger,
                client=httpx.Client(transport=httpx.MockTransport(upstream)),
            ).__enter__()
            with ThreadPoolExecutor(1) as pool:
                pending = pool.submit(self.request, proxy)
                deadline = time.monotonic() + 5
                while not proxy.budget.in_flight and time.monotonic() < deadline:
                    time.sleep(0.01)
                started = time.monotonic()
                proxy.__exit__(None, None, None)
                self.assertLess(time.monotonic() - started, 1)
                self.assertEqual(json.loads(ledger.read_text())["in_flight"], 1)
                self.assertFalse(proxy.client.is_closed)
                release.set()
                self.assertEqual(pending.result().status_code, 200)
            recorded = json.loads(ledger.read_text())
            self.assertEqual(recorded["in_flight"], 0)
            self.assertEqual(recorded["spent_usd"], "0.01")
            self.assertTrue(proxy.client.is_closed)

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
