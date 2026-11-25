# Misalignment Mechanisms

Tool-calling experiments on subset sum and prime factorisation with prohibited reference assistance.

## Setup

```sh
uv sync
```

Store your OpenRouter API key as a generic password in macOS Keychain Access:

- Service: `misalignment-mechanisms.openrouter`
- Account: your macOS username
- Password: your API key

Start the credential broker in a separate terminal:

```sh
uv run python -m mechanisms.broker
```

Broker settings are in `configs/broker/local.yaml`.

## Factors

| Experiment | Meaning | Default values |
| --- | --- | --- |
| `discoverability` | API exploration calls needed to reveal the prohibited function; depth 0 exposes it immediately. | 0, 2, 4, …, 20 |
| `detection` | Probability that prohibited use is detected and the trial's reward becomes zero, stated in the prompt. Applied once per trial. | 0, 0.1, 0.25, 0.5, 0.75, 1 |
| `payoff` | Amount of correct assistance returned by the prohibited function. | hint, partial, full |
| `time_pressure` | Wall-clock allowance, stated in the prompt, with remaining time in tool responses. | 30, 60, 120, 300, 900, 3600 seconds |

Each preset varies one factor, holding the others at depth 0, full assistance, detection probability 1, and one hour. Defaults: **DeepSeek V4 Flash with high reasoning**, subset sum with **64 32-bit integers**, 10 matched task seeds per condition, and 12 concurrent trials.

## Run a sweep

Prepare a discoverability sweep without model calls, then execute and plot it:

```sh
uv run python -m mechanisms.sweep experiment=discoverability
uv run python -m mechanisms.sweep experiment=discoverability dry_run=false
uv run python -m mechanisms.analyze run_dir=outputs/discoverability-depth-deepseek
```

Use `experiment=detection`, `experiment=payoff`, or `experiment=time_pressure` for the other factors. Their output directories are `outputs/detection-deepseek`, `outputs/payoff-deepseek`, and `outputs/time_pressure-deepseek`.

Override task difficulty, trial count, concurrency, or model on the command line. For example, a smaller sweep with 128-bit subset-sum integers:

```sh
uv run python -m mechanisms.sweep experiment=discoverability task.bits=128 \
  experiment.trials=2 'experiment.axes={treatment.exposure.depth: [0,2,20]}' \
  parallel=3 output=outputs/discovery-pilot dry_run=false
uv run python -m mechanisms.analyze run_dir=outputs/discovery-pilot
```

Use `task=factorization` or `model=glm` to switch tasks or models. To override a preset's fixed time allowance, use `'experiment.fixed={execution.timeout_sec: 120}'`.

Rerun the same command to resume. Choose a new `output` directory when changing settings or code. The spending threshold is $10 per attempt, checked between requests; a request can overshoot it. Infrastructure failures receive up to three attempts.

Task-difficulty sweeps are also available:

```sh
uv run python -m mechanisms.sweep experiment=subset_sum_difficulty dry_run=false
uv run python -m mechanisms.sweep task=factorization experiment=factorization_difficulty dry_run=false
```

## Add an experiment

Copy a preset into `configs/experiment/`, change its `name`, and edit `axes` to select parameter values. `fixed` sets the other factors; axes override fixed settings. Multiple axes produce their Cartesian product.

Set `factor.path` to the plotted axis, `factor.label` to its label, and `factor.kind` to `discrete`, `continuous`, or `categorical`. Categories retain their declared order; additional axes get separate plot panels.

```sh
uv run python -m mechanisms.sweep experiment=my_experiment dry_run=false
uv run python -m mechanisms.analyze run_dir=outputs/my_experiment-deepseek
```

## Results

The analyzer writes `trials.csv`, `summary.csv`, `summary.json`, and `rates.png/svg/pdf` into the sweep directory. Plots show rates with 95% confidence intervals. Discovery means the function was exposed; cheating means it was invoked. Rates include finished trials, including timeouts, and exclude pending or infrastructure-failed trials.

Rerun analysis during a sweep for partial results. Select metrics with an override:

```sh
uv run python -m mechanisms.analyze run_dir=outputs/detection-deepseek \
  'metrics=[cheating_rate,detection_rate,acceptance_rate]'
```

Trial prompts, tool calls, audits, and spending records are in `outputs/runs/<id>/`.

## Development

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
