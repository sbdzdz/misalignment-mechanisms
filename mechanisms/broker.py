import json
import os
import re
import signal
import socket
import socketserver
import subprocess
import threading
import time
from decimal import Decimal
from pathlib import Path

from mechanisms.config import cli
from mechanisms.proxy import BudgetProxy

ROOT = Path(__file__).resolve().parents[1]


def keychain_secret(config):
    result = subprocess.run(
        [
            "/usr/bin/security",
            "find-generic-password",
            "-s",
            config["keychain_service"],
            "-a",
            config["keychain_account"],
            "-w",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode or not result.stdout.strip():
        raise RuntimeError(
            "Keychain credential unavailable; unlock/authorize it in Keychain Access"
        )
    return result.stdout.strip()


def broker_request(payload, config):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(config["request_timeout_sec"] + 60)
        try:
            connection.connect(str(ROOT / config["socket"]))
        except OSError:
            raise RuntimeError(
                "Start uv run python -m mechanisms.broker before running experiments"
            ) from None
        connection.sendall(json.dumps(payload).encode() + b"\n")
        with connection.makefile("rb") as stream:
            line = stream.readline(65536)
    try:
        result = json.loads(line)
    except ValueError:
        raise RuntimeError("Broker closed the connection without a reply") from None
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


class Broker:
    def __init__(self, config, key, proxy_factory=BudgetProxy):
        self.config, self.key, self.proxy_factory = (config, key, proxy_factory)
        self.proxies = {}
        self.lock = threading.Lock()
        self.seen = set()

    def dispatch(self, request):
        run_id = request.get("run_id", "")
        if not re.fullmatch("[A-Za-z0-9_-]+", run_id):
            raise ValueError("Invalid run identifier")
        with self.lock:
            if request.get("action") == "close":
                item = self.proxies.get(run_id)
                if item is None or request.get("token") != item.token:
                    raise ValueError("Invalid run token")
                del self.proxies[run_id]
            elif request.get("action") == "open":
                if run_id in self.seen:
                    raise ValueError("Run already registered; use a new run identifier")
                model = request["model"]
                if model not in self.config["models"]:
                    raise ValueError("Model not allowed by broker")
                limit = Decimal(request["limit"])
                if not limit.is_finite() or not 0 < limit <= Decimal(
                    str(self.config["max_trial_cost_usd"])
                ):
                    raise ValueError("Invalid trial cost limit")
                ledger = Path(request["ledger"]).resolve()
                expected = (
                    ROOT / self.config["output_root"] / run_id / "budget.json"
                ).resolve()
                if ledger != expected or ledger.exists():
                    raise ValueError(
                        "Ledger must be a new trial under broker output_root"
                    )
                item = self.proxy_factory(
                    self.key,
                    limit,
                    ledger,
                    model=model,
                    expires_at=time.time() + self.config["trial_lifetime_sec"],
                    request_timeout_sec=self.config["request_timeout_sec"],
                    base_url=self.config["base_url"],
                )
                item.__enter__()
                self.proxies[run_id] = item
                self.seen.add(run_id)
                return dict(token=item.token, url=item.url)
            else:
                raise ValueError("Unknown broker action")
        item.__exit__(None, None, None)
        return {"closed": True}

    def close_expired(self, all_runs=False):
        with self.lock:
            expired = [
                k
                for k, v in self.proxies.items()
                if all_runs or time.time() >= v.expires_at
            ]
            items = [self.proxies.pop(k) for k in expired]
        for item in items:
            item.__exit__(None, None, None)


def serve(config):
    path = ROOT / config["socket"]
    if path.exists():
        raise RuntimeError(
            "Broker socket already exists; check for a running broker before removing it"
        )
    key = keychain_secret(config)
    broker = Broker(config, key)
    path = ROOT / config["socket"]
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.request.settimeout(10)
            try:
                data = self.rfile.readline(65536)
                result = broker.dispatch(json.loads(data))
            except Exception:
                result = {
                    "error": "Broker rejected request; check model, budget, run ID and ledger path"
                }
            self.wfile.write(json.dumps(result).encode() + b"\n")

    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads = True

    stop = threading.Event()

    def reap():
        while not stop.wait(15):
            broker.close_expired()

    def stop_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_signal)
    with Server(str(path), Handler) as server:
        os.chmod(path, 0o600)
        threading.Thread(target=reap, daemon=True).start()
        print(
            "Credential broker ready; upstream key remains in broker memory.",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            broker.close_expired(all_runs=True)
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    cli("credentials", lambda cfg: serve(cfg["broker"]))
