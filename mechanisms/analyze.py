import csv
from collections import defaultdict
from decimal import Decimal

from omegaconf import OmegaConf

from mechanisms.config import ROOT, cli
from mechanisms.storage import read, save


def summarize(config):
    directory = ROOT / config["run_dir"]
    manifest = read(directory / "manifest.json")
    groups = defaultdict(list)
    spent, unknown = Decimal(0), 0
    for trial in manifest["trials"]:
        values = tuple(
            OmegaConf.select(OmegaConf.create(trial["config"]), key)
            for key in config["group_by"]
        )
        for attempt in trial["attempts"]:
            budget = ROOT / attempt["output"] / "budget.json"
            if budget.exists():
                ledger = read(budget)
                spent += Decimal(ledger["spent_usd"])
                unknown += ledger["status"] == "cost_unknown"
        final = trial["attempts"][-1] if trial["attempts"] else None
        state = (
            read(ROOT / final["output"] / "audit.json")
            if final and final["status"] == "finished"
            else None
        )
        groups[values].append(state)
    rows = []
    for values, states in groups.items():
        complete = [s for s in states if s is not None]
        n = len(complete)
        rows.append(
            dict(zip(config["group_by"], values))
            | dict(
                planned=len(states),
                finished=n,
                cheating=sum(s["cheated"] for s in complete),
                accepted=sum(s["accepted"] for s in complete),
                answer_returned=sum(s["answer_returned"] for s in complete),
                mean_score=sum(s["score"] for s in complete) / n if n else None,
                cheating_rate=sum(s["cheated"] for s in complete) / n if n else None,
                acceptance_rate=sum(s["accepted"] for s in complete) / n if n else None,
            )
        )
    save(
        directory / "summary.json",
        dict(
            groups=rows,
            known_spending_usd=str(spent),
            attempts_with_unknown_cost=unknown,
        ),
    )
    with (directory / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if config["plot"] and any(row["finished"] for row in rows):
        plot(config, rows, directory)
    print(
        f"Finished: {sum(r['finished'] for r in rows)}/{sum(r['planned'] for r in rows)}; known spending: ${spent}; unknown-cost attempts: {unknown}"
    )


def plot(config, rows, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=tuple(config["figure_size"]), layout="constrained")
    labels = [
        " / ".join(str(row[key]) or config["empty_label"] for key in config["group_by"])
        for row in rows
    ]
    for metric, style in config["series"].items():
        ax.plot(
            range(len(rows)),
            [
                row[metric] * 100 if row[metric] is not None else float("nan")
                for row in rows
            ],
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
        )
    ax.set(
        xticks=range(len(rows)),
        xticklabels=labels,
        xlabel=config["xlabel"],
        ylabel=config["ylabel"],
        ylim=(0, 100),
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for extension in config["formats"]:
        fig.savefig(directory / f"rates.{extension}")
    plt.close(fig)


if __name__ == "__main__":
    cli("analysis", summarize)
