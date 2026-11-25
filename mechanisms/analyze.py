"""Export normalized trial data and plot any declared experiment factor."""

import csv
import math
from collections import defaultdict
from decimal import Decimal

from omegaconf import OmegaConf

from mechanisms.config import ROOT, cli
from mechanisms.storage import read, save

METRICS = {
    "cheating_rate": "cheated",
    "discovery_rate": "shortcut_discovered",
    "acceptance_rate": "accepted",
    "detection_rate": "detected",
    "assistance_rate": "assistance_delivered",
    "timeout_rate": "timed_out",
}


def interval(successes, n):
    """95% Wilson interval, undefined when there are no observations."""
    if not n:
        return None, None
    p, z = successes / n, 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0, center - half), min(1, center + half)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(config):
    metrics = config.get("metrics", [])
    if config["plot"] and (
        not metrics
        or (
            any(m in METRICS for m in metrics)
            and not all(m in METRICS for m in metrics)
        )
    ):
        raise ValueError(
            "Choose rate metrics or mean metrics in a single plot, not both"
        )
    directory = ROOT / config["run_dir"]
    manifest = read(directory / "manifest.json")
    experiment = manifest["config"]["experiment"]
    axes = experiment["axes"]
    default_path = next(iter(axes))
    factor = dict(
        experiment.get("factor")
        or dict(path=default_path, label=default_path, kind="auto")
    )
    if config.get("factor"):
        factor = dict(
            path=config["factor"],
            label=config.get("xlabel") or config["factor"],
            kind=config.get("kind", "auto"),
        )
    factor["label"] = config.get("xlabel") or factor["label"]
    if factor["path"] not in axes:
        raise ValueError("Plot factor must be a declared sweep axis")
    if factor["kind"] == "auto":
        factor["kind"] = (
            "discrete"
            if all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in axes[factor["path"]]
            )
            else "categorical"
        )
    if factor["kind"] not in {"discrete", "continuous", "categorical"}:
        raise ValueError("Unknown factor kind")
    group_by = config.get("group_by") or list(axes)
    if factor["path"] not in group_by or not set(axes).issubset(group_by):
        raise ValueError(
            "group_by must include every varied axis to avoid pooling conditions"
        )
    groups = defaultdict(list)
    trial_rows = []
    spent, unknown = Decimal(0), 0
    for trial in manifest["trials"]:
        spec = OmegaConf.create(trial["config"])
        values = {key: OmegaConf.select(spec, key) for key in group_by}
        cost, missing = Decimal(0), 0
        for attempt in trial["attempts"]:
            budget = ROOT / attempt["output"] / "budget.json"
            if budget.exists():
                ledger = read(budget)
                cost += Decimal(ledger["spent_usd"])
                missing += ledger["status"] == "cost_unknown" or bool(
                    attempt.get("cost_pending")
                )
            else:
                missing += 1
        spent += cost
        unknown += missing
        final = trial["attempts"][-1] if trial["attempts"] else None
        state = (
            read(ROOT / final["output"] / "audit.json")
            if final and (ROOT / final["output"] / "audit.json").exists()
            else {}
        )
        finished = bool(final and final["status"] == "finished")
        row = dict(
            trial_id=trial["id"],
            seed=trial["instance"]["seed"],
            **values,
            status=final["status"] if final else "pending",
            outcome=final.get("outcome") if final else None,
            finished=finished,
            attempts=len(trial["attempts"]),
            known_spending_usd=str(cost),
            attempts_with_unknown_cost=missing,
        )
        for field in [
            *dict.fromkeys(METRICS.values()),
            "answer_returned",
            "score",
            "elapsed_seconds",
            "exploration_calls",
            "deepest_namespace",
            "assistance_level",
        ]:
            row[field] = state.get(field)
        row["timed_out"] = row["outcome"] == "timeout" if finished else None
        trial_rows.append(row)
        groups[tuple(values.values())].append(row)
    rows = []
    for values, trials in groups.items():
        complete = [t for t in trials if t["finished"]]
        n = len(complete)
        row = dict(zip(group_by, values)) | dict(
            planned=len(trials),
            finished=n,
            failed=sum(t["status"] == "failed" for t in trials),
        )
        for status in ("pending", "running", "interrupted"):
            row[status] = sum(t["status"] == status for t in trials)
        for metric, field in METRICS.items():
            observed = [t[field] for t in complete if t[field] is not None]
            count = sum(bool(v) for v in observed)
            low, high = interval(count, len(observed))
            row.update(
                {
                    metric: count / len(observed) if observed else None,
                    metric + "_count": count,
                    metric + "_n": len(observed),
                    metric + "_low": low,
                    metric + "_high": high,
                }
            )
        discovered = [t for t in complete if t["shortcut_discovered"]]
        row["cheating_given_discovery"] = (
            sum(bool(t["cheated"]) for t in discovered) / len(discovered)
            if discovered
            else None
        )
        row["discovered_n"] = len(discovered)
        for field in ("score", "elapsed_seconds", "exploration_calls"):
            observed = [t[field] for t in complete if t[field] is not None]
            row["mean_" + field] = sum(observed) / len(observed) if observed else None
        rows.append(row)
    summary = dict(
        schema_version=2,
        factor=factor,
        group_by=group_by,
        groups=rows,
        known_spending_usd=str(spent),
        attempts_with_unknown_cost=unknown,
        denominator="Finished trials, including timeouts and resource limits; excludes infrastructure failures and pending trials.",
    )
    save(directory / "summary.json", summary)
    write_csv(directory / "summary.csv", rows)
    write_csv(directory / "trials.csv", trial_rows)
    if config["plot"] and any(row["finished"] for row in rows):
        plot(config, rows, directory, factor, axes[factor["path"]], group_by)
    print(
        f"Finished: {sum(r['finished'] for r in rows)}/{sum(r['planned'] for r in rows)}; known spending: ${spent}; attempts with unknown/pending costs: {unknown}"
    )
    return summary


def plot(config, rows, directory, factor, order, group_by):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    others = [key for key in group_by if key != factor["path"]]
    panels = defaultdict(list)
    for row in rows:
        panels[tuple(row[key] for key in others)].append(row)
    palette = ["#8172B3", "#5C9EAD", "#D69A76", "#80A889"]
    with plt.rc_context(
        {
            "font.size": 11,
            "text.color": "#444454",
            "axes.labelcolor": "#444454",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(
            len(panels),
            1,
            figsize=(config["figure_size"][0], config["figure_size"][1] * len(panels)),
            squeeze=False,
            layout="constrained",
        )
        fig.patch.set_facecolor("white")
        categorical = factor["kind"] == "categorical"
        order = list(order) if categorical else sorted(order)
        for ax, (panel, entries) in zip(axes[:, 0], panels.items()):
            entries.sort(key=lambda r: order.index(r[factor["path"]]))
            x = [
                order.index(r[factor["path"]]) if categorical else r[factor["path"]]
                for r in entries
            ]
            for index, metric in enumerate(config["metrics"]):
                if metric not in METRICS and metric not in {
                    "mean_score",
                    "mean_elapsed_seconds",
                    "mean_exploration_calls",
                }:
                    raise ValueError(f"Unknown metric: {metric}")
                scale = 100 if metric in METRICS else 1
                y = [
                    row[metric] * scale if row[metric] is not None else float("nan")
                    for row in entries
                ]
                ax.plot(
                    x,
                    y,
                    color=palette[index % len(palette)],
                    marker="o",
                    markersize=7,
                    markeredgewidth=0,
                    linewidth=2,
                    linestyle="none" if categorical else "-",
                    label=metric.replace("_", " ").capitalize(),
                    zorder=3,
                )
            ax.set_xticks(x, [str(row[factor["path"]]) for row in entries])
            ax.set_xlabel(factor["label"], labelpad=12)
            rate_only = all(metric in METRICS for metric in config["metrics"])
            label = (
                config["metrics"][0].replace("_", " ").capitalize()
                if len(config["metrics"]) == 1
                else "Rate"
                if rate_only
                else "Value"
            )
            ax.set_ylabel(label, labelpad=12)
            if rate_only:
                ax.set_ylim(-3, 103)
                ax.set_yticks(range(0, 101, 20))
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            if others:
                ax.set_title(
                    " / ".join(f"{key}={value}" for key, value in zip(others, panel)),
                    loc="left",
                    pad=16,
                )
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor("white")
            ax.set_axisbelow(True)
            ax.grid(axis="both", color="#E9E7EF", linewidth=0.7)
            ax.tick_params(axis="both", which="both", length=0, pad=8, colors="#6C6B7A")
            ax.margins(x=0.04)
            if len(config["metrics"]) > 1:
                ax.legend(frameon=False)
        for extension in config["formats"]:
            fig.savefig(
                directory / f"rates.{extension}",
                dpi=200,
                facecolor="white",
                edgecolor="none",
            )
        plt.close(fig)


if __name__ == "__main__":
    cli("analysis", summarize)
