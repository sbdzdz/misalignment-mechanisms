import json
import secrets
import threading
import time
from decimal import Decimal, InvalidOperation
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from mechanisms.model_log import ModelLog, Response


class Budget:
    def __init__(self, limit, path):
        self.limit = Decimal(str(limit))
        if not self.limit.is_finite() or self.limit <= 0:
            raise ValueError("Budget must be a finite positive dollar amount")
        self.path = Path(path)
        self.spent = Decimal(0)
        self.unknown = False
        self.requests = []
        self.in_flight = 0
        self.lock = threading.Lock()
        self.save()

    @property
    def status(self):
        if self.unknown:
            return "cost_unknown"
        return "budget_exhausted" if self.spent >= self.limit else "available"

    def record(self, cost, generation_id=None, **metadata):
        try:
            value = Decimal(str(cost))
            if not value.is_finite() or value < 0:
                raise ValueError("Invalid cost")
        except (InvalidOperation, ValueError):
            value = None
        if value is None:
            self.unknown = True
        else:
            self.spent += value
        self.requests.append(
            {
                "generation_id": generation_id,
                "cost_usd": str(value) if value is not None else None,
                **metadata,
            }
        )
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "limit_usd": str(self.limit),
                    "spent_usd": str(self.spent),
                    "status": self.status,
                    "in_flight": self.in_flight,
                    "requests": self.requests,
                },
                indent=2,
            )
        )
        temporary.replace(self.path)


class BudgetProxy:
    def __init__(
        self,
        key,
        limit,
        path,
        *,
        client=None,
        log_path=None,
        request_timeout_sec=600,
        model=None,
        expires_at=None,
        base_url="https://openrouter.ai/api/v1",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.expires_at = expires_at
        self.budget = Budget(limit, path)
        self.log = ModelLog(log_path or Path(path).with_name("model-calls.jsonl"))
        self.token = secrets.token_urlsafe(32)
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(request_timeout_sec, connect=30), trust_env=False
        )
        self.key = key
        self.stopping = False
        # Separate from budget.lock, which is held for a whole request.
        self.state = threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def error(self, status, message):
                data = json.dumps(
                    {"error": {"message": message, "type": "run_budget"}}
                ).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
                self.close_connection = True

            def do_POST(self):
                if self.headers.get("Authorization") != f"Bearer {proxy.token}":
                    self.error(401, "Invalid run token")
                    return
                if self.path != "/v1/chat/completions":
                    self.error(404, "Only chat completions are supported")
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 32 * 1024 * 1024:
                        raise ValueError()
                    payload = json.loads(self.rfile.read(size))
                    if not isinstance(payload, dict):
                        raise ValueError()
                except (ValueError, json.JSONDecodeError):
                    self.error(400, "Invalid request")
                    return
                with proxy.budget.lock:
                    if proxy.expires_at is not None and time.time() >= proxy.expires_at:
                        self.error(403, "Run token expired")
                        return
                    if proxy.model is not None and payload.get("model") != proxy.model:
                        self.error(403, "Model not permitted for this run")
                        return
                    if proxy.budget.status != "available":
                        self.error(402, proxy.budget.status)
                        return
                    if not proxy.begin():
                        self.error(503, "Run has ended")
                        return
                    try:
                        self.forward(payload)
                    finally:
                        proxy.end()

            def forward(self, payload):
                cost, generation = (None, None)
                transport_error = None
                status = None
                parsed = Response()
                began = time.time()
                started = False
                downstream_alive = True

                def observe(data):
                    nonlocal cost, generation
                    parsed.add(data)
                    if isinstance(data, dict):
                        generation = data.get("id") or generation
                        usage = data.get("usage")
                        if isinstance(usage, dict) and usage.get("cost") is not None:
                            cost = usage["cost"]

                def write(data):
                    nonlocal downstream_alive
                    if downstream_alive:
                        try:
                            self.wfile.write(data)
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            downstream_alive = False

                try:
                    with proxy.client.stream(
                        "POST",
                        proxy.base_url + "/chat/completions",
                        headers={
                            "Authorization": f"Bearer {proxy.key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    ) as response:
                        status = response.status_code
                        self.send_response(response.status_code)
                        self.send_header(
                            "Content-Type",
                            response.headers.get("content-type", "application/json"),
                        )
                        self.send_header("Connection", "close")
                        self.end_headers()
                        started = True
                        if "text/event-stream" in response.headers.get(
                            "content-type", ""
                        ):
                            for line in response.iter_lines():
                                if line.startswith("data:"):
                                    try:
                                        observe(json.loads(line[5:].strip()))
                                    except ValueError:
                                        pass
                                write((line + "\n").encode())
                        else:
                            data = response.read()
                            try:
                                observe(json.loads(data))
                            except ValueError:
                                pass
                            write(data)
                except Exception as exc:
                    transport_error = type(exc).__name__
                    if not started:
                        self.error(502, "Upstream request failed; cost unknown")
                finally:
                    source = "stream_or_response"
                    if cost is None and generation:
                        cost = proxy.lookup_cost(generation)
                        source = (
                            "generation_lookup" if cost is not None else "unresolved"
                        )
                    proxy.budget.record(
                        cost,
                        generation,
                        cost_source=source,
                        transport_error=transport_error,
                    )
                    try:
                        proxy.log.record(
                            payload,
                            parsed,
                            started_at=began,
                            finished_at=time.time(),
                            status=status,
                            generation_id=generation,
                            cost_usd=str(cost) if cost is not None else None,
                            transport_error=transport_error,
                        )
                    except Exception as exc:
                        proxy.log.write({"type": "log_error", "error": repr(exc)})
                    self.close_connection = True

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def lookup_cost(self, generation):
        for attempt in range(3):
            if attempt:
                time.sleep(attempt)
            try:
                response = self.client.get(
                    self.base_url + "/generation",
                    params={"id": generation},
                    headers={"Authorization": f"Bearer {self.key}"},
                    timeout=10,
                )
                response.raise_for_status()
                value = response.json().get("data", {}).get("total_cost")
                cost = Decimal(str(value))
                if cost.is_finite() and cost >= 0:
                    return cost
            except (httpx.HTTPError, ValueError, InvalidOperation, AttributeError):
                pass
        return None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def begin(self):
        """Register an upstream request unless the run has ended.

        Returns:
            True if the request may proceed.
        """
        with self.state:
            if self.stopping:
                return False
            self.budget.in_flight += 1
            self.budget.save()
            return True

    def end(self):
        """Unregister a request; release the client once closed and idle."""
        with self.state:
            self.budget.in_flight -= 1
            self.budget.save()
            release = self.stopping and not self.budget.in_flight
        if release:
            self.client.close()

    def __exit__(self, *args):
        """Stop serving; requests in flight finish and record their cost.

        Args:
            args: Exception type, value, and traceback; unused.
        """
        with self.state:
            self.stopping = True
            idle = not self.budget.in_flight
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        if idle:
            self.client.close()
