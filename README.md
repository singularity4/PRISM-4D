# PRISM-4D
A Benchmark for Probabilistic Moral and Safety Reasoning in LLMs, based on the [Moral Machine experiment](https://doi.org/10.1038/s41586-018-0637-6) by Awad et al. (2018) at MIT Media Lab. PRISM-4D extends smart car scenarios to probabilistic outcomes and introduces value-aligned CoT. We propose four evaluation metrics for AI safety: Bayes regret, stochastic dominance, risk attitude and counterfactual sensitivity.

## 4D Evaluation metrics

`evaluation_metrics.py` computes four metrics from model `responses.jsonl` and generated `scenarios.jsonl`:

1. Bayes regret / Expected-harm regret (decision theory)
2. Risk attitude (decision theory)
3. Dominance violation rate (stochastic dominance)
4. Counterfactual sensitivity (counterfactual fairness)

AMCE per dimension is conjoint analysis based on Moral Machine.

```
python scripts/evaluation_metrics.py \
    --scenarios data/scenarios.jsonl \
    --responses data/responses.jsonl \
    --out data/metrics.json
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
