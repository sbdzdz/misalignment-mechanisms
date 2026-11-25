import copy
import json

import pytest

from mechanisms import analyze
from mechanisms.storage import save


@pytest.mark.parametrize(
    "kind, values",
    [("discrete", [0, 2, 20]), ("categorical", ["hint", "partial", "full"])],
)
def test_analysis_exports_counts_and_plots(tmp_path, monkeypatch, kind, values):
    monkeypatch.setattr(analyze, "ROOT", tmp_path)
    directory = tmp_path / "sweep"
    key = "treatment.value"
    experiment = dict(
        axes={key: values}, factor=dict(path=key, label="Test factor", kind=kind)
    )
    trials = []
    for i, value in enumerate(values):
        output = f"runs/{i}"
        state = dict(
            cheated=i == 0,
            shortcut_discovered=True,
            accepted=i == 0,
            detected=False,
            assistance_delivered=i == 0,
            score=int(i == 0),
            elapsed_seconds=1,
            exploration_calls=0,
        )
        save(tmp_path / output / "audit.json", state)
        save(tmp_path / output / "budget.json", dict(spent_usd="0.01", status="active"))
        trials.append(
            dict(
                id=str(i),
                config=dict(treatment=dict(value=value)),
                instance=dict(seed=0),
                attempts=[
                    dict(
                        output=output,
                        status="finished",
                        outcome="timeout" if i else "accepted",
                    )
                ],
            )
        )
    failed = copy.deepcopy(trials[0])
    failed["id"] = "failed"
    failed["attempts"][0]["status"] = "failed"
    trials.append(failed)
    save(
        directory / "manifest.json",
        dict(config=dict(experiment=experiment), trials=trials),
    )
    summary = analyze.summarize(
        dict(
            run_dir="sweep",
            plot=True,
            figure_size=[8, 4],
            formats=["png", "svg", "pdf"],
            metrics=["cheating_rate"],
        )
    )
    assert summary["groups"][0]["planned"] == 2
    assert summary["groups"][0]["finished"] == 1
    assert summary["groups"][0]["cheating_rate"] == 1
    assert summary["groups"][1]["timeout_rate"] == 1
    for file in ("summary.csv", "trials.csv", "rates.png", "rates.svg", "rates.pdf"):
        assert (directory / file).stat().st_size > 0
    assert (
        json.loads((directory / "summary.json").read_text())["factor"]["kind"] == kind
    )
