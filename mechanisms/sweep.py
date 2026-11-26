import asyncio
import copy
import fcntl
import itertools
import random
import time
import uuid

import httpx
from omegaconf import OmegaConf

from mechanisms.broker import broker_request
from mechanisms.config import ROOT, cli
from mechanisms.environment import TASKS, Environment
from mechanisms.runner import ProviderError, run
from mechanisms.storage import digest, read, save


def expand(config):
    axes = config["experiment"]["axes"]
    if (
        not axes
        or any(not values for values in axes.values())
        or config["experiment"]["trials"] < 1
    ):
        raise ValueError("Provide nonempty axes and a positive trial count")
    for values in itertools.product(*axes.values()):
        spec = OmegaConf.create(
            {
                k: config[k]
                for k in ("task", "model", "treatment", "environment", "execution")
            }
        )
        setting = dict(zip(axes, values))
        missing = object()
        for key, value in (config["experiment"].get("fixed", {}) | setting).items():
            if OmegaConf.select(spec, key, default=missing) is missing:
                raise ValueError(f"Unknown sweep axis: {key}")
            OmegaConf.update(spec, key, value)
        resolved = OmegaConf.to_container(spec, resolve=True)
        for seed in range(
            config["experiment"]["seed_start"],
            config["experiment"]["seed_start"] + config["experiment"]["trials"],
        ):
            yield copy.deepcopy(resolved), seed, setting


def prepare(config, directory):
    specifications = list(expand(config))
    source = {
        str(p.relative_to(ROOT)): p.read_text()
        for p in sorted((ROOT / "mechanisms").rglob("*.py"))
    }
    lock = ROOT / "uv.lock"
    if lock.exists():
        source["uv.lock"] = lock.read_text()
    fingerprint = digest(
        dict(
            specifications=specifications,
            source=source,
            experiment=config["experiment"],
        )
    )
    path = directory / "manifest.json"
    if path.exists():
        manifest = read(path)
        if manifest["fingerprint"] != fingerprint:
            raise ValueError(
                "Run configuration or implementation changed; choose a new output directory"
            )
        return manifest
    trials = []
    for spec, seed, axes in specifications:
        public, answer = TASKS[spec["task"]["name"]].generate(spec["task"], seed)
        instance = dict(seed=seed, public=public, answer=answer)
        identifier = digest(dict(config=spec, instance=instance))[:20]
        trials.append(
            dict(id=identifier, config=spec, instance=instance, axes=axes, attempts=[])
        )
    if len({t["id"] for t in trials}) != len(trials):
        raise ValueError("Sweep contains duplicate trials")
    manifest = dict(fingerprint=fingerprint, config=config, trials=trials)
    save(directory / "source.json", source)
    save(path, manifest)
    return manifest


async def execute(config, directory, manifest):
    semaphore = asyncio.Semaphore(config["parallel"])

    def persist():
        save(directory / "manifest.json", manifest)

    async def attempt(trial):
        async with semaphore:
            run_id = uuid.uuid4().hex
            folder = ROOT / config["broker"]["output_root"] / run_id
            folder.mkdir(parents=True)
            record = dict(
                id=run_id,
                output=str(folder.relative_to(ROOT)),
                status="running",
                started_at=time.time(),
            )
            trial["attempts"].append(record)
            persist()
            save(folder / "config.json", trial["config"])
            save(folder / "instance.json", trial["instance"])
            env = Environment(trial["config"], trial["instance"], folder / "audit.json")
            (folder / "prompt.txt").write_text(env.instruction)
            save(folder / "tools.json", env.catalog)
            credentials = None
            try:
                credentials = await asyncio.to_thread(
                    broker_request,
                    dict(
                        action="open",
                        run_id=run_id,
                        model=trial["config"]["model"]["id"],
                        limit=str(trial["config"]["execution"]["max_cost_usd"]),
                        ledger=str(folder / "budget.json"),
                    ),
                    config["broker"],
                )
                outcome = await run(trial["config"], env, credentials)
                record.update(status="finished", outcome=outcome)
            except asyncio.CancelledError:
                record.update(status="interrupted", outcome="interrupted")
                raise
            except httpx.HTTPStatusError as error:
                retryable = (
                    error.response.status_code in {408, 429}
                    or error.response.status_code >= 500
                )
                record.update(
                    status="failed",
                    retryable=retryable,
                    error=f"HTTP {error.response.status_code}",
                )
            except (
                httpx.TransportError,
                ProviderError,
                KeyError,
                IndexError,
                ValueError,
            ) as error:
                record.update(
                    status="failed", retryable=True, error=type(error).__name__
                )
            except RuntimeError as error:
                record.update(status="failed", retryable=False, error=str(error))
            finally:
                env.finish(record.get("outcome", record["status"]))
                if credentials:
                    try:
                        await asyncio.to_thread(
                            broker_request,
                            dict(
                                action="close",
                                run_id=run_id,
                                token=credentials["token"],
                            ),
                            config["broker"],
                        )
                    except (RuntimeError, OSError):
                        record["cost_pending"] = True
                record["finished_at"] = time.time()
                save(folder / "result.json", record)
                persist()
                print(
                    f"{trial['axes']} seed={trial['instance']['seed']} {record['status']} {record.get('outcome', record.get('error'))}",
                    flush=True,
                )

    async def slot(trial):
        if trial["attempts"] and trial["attempts"][-1]["status"] == "running":
            trial["attempts"][-1]["status"] = "interrupted"
        while len(trial["attempts"]) < config["max_attempts"]:
            if trial["attempts"]:
                last = trial["attempts"][-1]
                if last["status"] == "finished" or (
                    last["status"] == "failed" and not last.get("retryable")
                ):
                    return
                await asyncio.sleep(config["retry_delay_sec"])
            await attempt(trial)

    ordered = list(manifest["trials"])
    random.Random(config["experiment"].get("schedule_seed", 0)).shuffle(ordered)
    await asyncio.gather(*(slot(trial) for trial in ordered))
    persist()


def main(config):
    if config["parallel"] < 1 or config["max_attempts"] < 1:
        raise ValueError("parallel and max_attempts must be positive")
    directory = ROOT / config["output"]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare(config, directory)
        print(f"{len(manifest['trials'])} trials: {directory}")
        if not config["dry_run"]:
            asyncio.run(execute(config, directory, manifest))


if __name__ == "__main__":
    cli("sweep", main)
