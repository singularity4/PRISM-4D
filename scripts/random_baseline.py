"""
Random-choice baseline for PRISM-4D (planned in docs/design.md).

Writes responses.jsonl rows for a "model" named random_baseline that
picks Action 1 or Action 2 with equal probability (seeded), so it can be
scored by evaluation_metrics.py like any model.

With --reference-seeds N it also writes the random-choice reference: for
each seed 0, ..., N-1, one random run (drawn as above) is scored with
evaluate_model from evaluation_metrics.py, and the mean and the 95% range
(2.5th to 97.5th percentile) of each metric over the N runs are written to
--reference-out.

    python scripts/random_baseline.py --scenarios data/scenarios_1000.jsonl \
        --out data/random_baseline_responses.jsonl --seed 0
    python scripts/random_baseline.py --scenarios data/scenarios_1000.jsonl \
        --out data/random_baseline_responses.jsonl --seed 0 \
        --reference-seeds 1000 --reference-out data/random_reference.json
"""

import argparse
import json
import random

from evaluation_metrics import evaluate_model, load_scenarios


def summarize_over_seeds(values):
    """Mean and 95% range of one metric over the random runs."""
    ordered = sorted(value for value in values if value is not None)
    return {"mean": sum(ordered) / len(ordered),
            "range": [ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered)) - 1]]}


def random_reference(scenarios_path, n_seeds):
    """Random-choice reference: for each seed, one random run scored with evaluate_model."""
    scenarios = load_scenarios(scenarios_path)
    runs = []
    for seed in range(n_seeds):
        random_generator = random.Random(seed)
        choices = [(scenario_id, random_generator.choice([1, 2])) for scenario_id in scenarios]
        runs.append(evaluate_model(scenarios, choices))
    reference = {"n_seeds": n_seeds}
    for metric in ("mean_expected_harm_regret", "risk_attitude_gamma",
                   "dominance_violation_rate", "mean_counterfactual_sensitivity"):
        reference[metric] = summarize_over_seeds([run[metric] for run in runs])
    for metric in ("counterfactual_sensitivity", "amce_per_dimension"):
        for dimension in runs[0][metric]:
            reference[f"{metric}.{dimension}"] = summarize_over_seeds([run[metric][dimension] for run in runs])
    return reference


def main():
    parser = argparse.ArgumentParser(description="Random-choice baseline.")
    parser.add_argument("--scenarios", default="data/scenarios.jsonl")
    parser.add_argument("--out", default="data/random_baseline_responses.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-seeds", type=int, default=0,
                        help="number of seeds for the random-choice reference; 0 disables (default: 0)")
    parser.add_argument("--reference-out", default="data/random_reference.json")
    arguments = parser.parse_args()

    random_generator = random.Random(arguments.seed)
    with open(arguments.scenarios) as scenario_file, open(arguments.out, "w") as responses_file:
        for line in scenario_file:
            scenario = json.loads(line)
            responses_file.write(json.dumps({"model_name": "random_baseline",
                                             "scenario_id": scenario["id"],
                                             "chosen_action": random_generator.choice([1, 2])}) + "\n")
    print(f"Wrote random baseline to {arguments.out}")

    if arguments.reference_seeds > 0:
        reference = random_reference(arguments.scenarios, arguments.reference_seeds)
        with open(arguments.reference_out, "w") as reference_file:
            json.dump(reference, reference_file, indent=2)
        print(f"Wrote random-choice reference over {arguments.reference_seeds} seeds to {arguments.reference_out}")


if __name__ == "__main__":
    main()
