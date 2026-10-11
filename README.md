# PRISM-4D
A Benchmark for Probabilistic Moral and Safety Reasoning in LLMs, based on the [Moral Machine experiment](https://doi.org/10.1038/s41586-018-0637-6) by Awad et al. (2018) at MIT Media Lab. PRISM-4D extends smart car scenarios to probabilistic outcomes and introduces four evaluation metrics for AI safety: Bayes regret, stochastic dominance, risk attitude and counterfactual sensitivity.

## 4D Evaluation metrics

`evaluation_metrics.py` computes four metrics from model `responses.jsonl` and generated `scenarios.jsonl`:

1. Bayes regret / Expected-harm regret (decision theory)
2. Risk attitude (decision theory)
3. Dominance violation rate (stochastic dominance)
4. Counterfactual sensitivity (counterfactual fairness)

AMCE per dimension is conjoint analysis based on Moral Machine.
Every metric is reported with a 95% bootstrap interval (2,000 resamples over scenarios; `--bootstrap`, `--seed`).

```
python scripts/evaluation_metrics.py \
    --scenarios data/scenarios.jsonl \
    --responses data/responses.jsonl \
    --out data/metrics.json
```

Metric 4 (counterfactual sensitivity) from twin scenarios: `twin_scenarios.py` creates, for each scenario that targets
species, social status, gender, age, fitness or lawfulness, a twin that differs only in the targeted attribute.
`counterfactual_sensitivity_twins.py` computes the share of scenarios answered differently from their twins, with
random choice as reference. It takes one responses file per run for the scenarios and for the twins; `{model}` in a path is
replaced by each model name, and run k of the scenarios is paired with run k of the twins.

```
python scripts/twin_scenarios.py --scenarios data/scenarios_1000.jsonl --out data/scenarios_twins.jsonl
python scripts/counterfactual_sensitivity_twins.py --models haiku sonnet opus \
    --scenario-responses data/responses_{model}_1000.jsonl data/responses_{model}_trial2.jsonl data/responses_{model}_trial3.jsonl \
    --twin-responses data/responses_{model}_twins.jsonl data/responses_{model}_twins2.jsonl data/responses_{model}_twins3.jsonl
```

`max_likelihood.py` fits metric 2 (risk attitude gamma) by maximum likelihood, with a logistic choice rule, as a
complement to the grid fit in `evaluation_metrics.py`. It needs numpy and scipy (`pip install -r requirements.txt`).

```
python scripts/max_likelihood.py --scenarios data/scenarios.jsonl --responses data/responses.jsonl --out data/max_likelihood.json
```

## Design

See `docs/design.md` for the full scenario design: 10 scenario dimensions,
3 trade-off types, character pool, probability range, and
the future extensions.

## Quick start

Generate 100 scenarios with the default seed:

```
python scripts/generate.py --n 100 --out data/scenarios.jsonl
```
CLI flags:

- `--n`: number of scenarios (default 100).
- `--out`: output path (default `data/scenarios.jsonl`).
- `--seed`: master seed (default 0). Same seed gives byte-identical output.

Each line of `scenarios.jsonl` is a JSON object:

```
{
  "id": <int>,
  "prompt": <text shown to the model>,
  "scenario_info": {
    "dimension": <one of 10 moral dimensions>,
    "tradeoff_type": <one of 3 trade-off types>,
    "group_left": [...],
    "group_right": [...],
    "probability_left": <int %>,
    "probability_right": <int %>,
    "probability_action_1": <int %>,
    "probability_action_2": <int %>,
    "n_action_1": <int>,
    "n_action_2": <int>,
    "group_action_1": [...],
    "group_action_2": [...],
    "is_lawful_left": <bool>,
    "is_lawful_right": <bool>,
    "left_is_passengers": <bool>,
    "right_is_passengers": <bool>,
    "slot_1_action": "stay" | "swerve",
    "slot_2_action": "stay" | "swerve",
    "slots_swapped": <bool>
  }
}
```

## Running models

`orchestrate.py` does not call a model. It splits scenarios into batches, parses each model's answers into
`responses.jsonl` and records SHA-256 hashes, so any model can answer the same batches.
`random_baseline.py` writes responses of uniform random choice in the same format. With `--reference-seeds 1000` it
also writes the random-choice reference: the mean and 95% range of each metric over 1,000 random runs.

```
python scripts/orchestrate.py prepare --scenarios data/scenarios_1000.jsonl --out data/batches --batch-size 150 --seed 0
python scripts/orchestrate.py parse --batch data/batches/batch_000.json --answer <model reply> --model <name> --out data/responses.jsonl
python scripts/orchestrate.py hash --file data/responses.jsonl
python scripts/random_baseline.py --scenarios data/scenarios_1000.jsonl --out data/random_baseline_responses.jsonl --seed 0
python scripts/random_baseline.py --scenarios data/scenarios_1000.jsonl --out data/random_baseline_responses.jsonl --seed 0 --reference-seeds 1000 --reference-out data/random_reference.json
```

`data/scenarios_1000.jsonl` holds the 1,000 scenarios used in the paper (`generate.py --n 1000`); its first 100 equal
`data/scenarios.jsonl`.
