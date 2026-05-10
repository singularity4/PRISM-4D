"""
Dataset statistics for the PRISM-4D benchmark.

================================================================================
SCRIPT DESCRIPTION
================================================================================
Reads scenarios.jsonl and reports descriptive statistics about the
benchmark. 

Reports:
  - Scenarios: count, dimension distribution, trade-off type
    distribution, joint (dimension, trade-off type) distribution.
  - Probability and group size: marginal distributions per side and
    differences between sides.
  - Action structure: stay/swerve slot assignment, swap rate,
    passenger/pedestrian distribution, lawfulness distribution.
  - Character usage: count and frequency of each of the 20 character
    types across all scenarios.
  - Derived quantities: expected harm (probability x count) per side,
    dominance-relevant scenario count, equal-expected-harm scenario
    count.

================================================================================
INPUT
================================================================================
A scenarios.jsonl file (default: ../data/scenarios.jsonl) produced by
generate.py.

================================================================================
USAGE
================================================================================
    python analyze_dataset_stats.py
    python analyze_dataset_stats.py --scenarios path/to/scenarios.jsonl
"""

import argparse
import json
from collections import Counter


def histogram(values):
    """Return a dict {value: count} sorted by value."""
    return dict(sorted(Counter(values).items()))


# ============================================================================
# SCENARIO-LEVEL
# ============================================================================

def report_scenario_stats(scenarios):
    print("\n=== SCENARIO-LEVEL ===")
    n = len(scenarios)
    print(f"Total scenarios: {n}")

    dimensions = [s["scenario_info"]["dimension"] for s in scenarios]
    print(f"\nDimension distribution:")
    for dimension, count in sorted(Counter(dimensions).items()):
        print(f"  {dimension:<18} {count:>5} ({count/n:.1%})")

    tradeoff_types = [s["scenario_info"]["tradeoff_type"] for s in scenarios]
    print(f"\nTrade-off type distribution:")
    for tradeoff_type, count in sorted(Counter(tradeoff_types).items()):
        print(f"  {tradeoff_type:<28} {count:>5} ({count/n:.1%})")

    joint = Counter((s["scenario_info"]["dimension"],
                     s["scenario_info"]["tradeoff_type"]) for s in scenarios)
    print(f"\nJoint (dimension, trade-off type) distribution:")
    for (dimension, tradeoff_type), count in sorted(joint.items()):
        print(f"  ({dimension:<18}, {tradeoff_type:<28}) {count:>4}")


# ============================================================================
# PROBABILITY AND GROUP SIZE
# ============================================================================

def report_probability_and_size_stats(scenarios):
    print("\n=== PROBABILITY AND GROUP SIZE ===")

    probabilities_action_1 = [s["scenario_info"]["probability_action_1"] for s in scenarios]
    probabilities_action_2 = [s["scenario_info"]["probability_action_2"] for s in scenarios]
    n_action_1 = [s["scenario_info"]["n_action_1"] for s in scenarios]
    n_action_2 = [s["scenario_info"]["n_action_2"] for s in scenarios]

    print(f"\nProbability distribution (Action 1):")
    for v, c in sorted(Counter(probabilities_action_1).items()):
        print(f"  {v}% : {c}")
    print(f"\nProbability distribution (Action 2):")
    for v, c in sorted(Counter(probabilities_action_2).items()):
        print(f"  {v}% : {c}")

    print(f"\nGroup size distribution (Action 1): {histogram(n_action_1)}")
    print(f"Group size distribution (Action 2): {histogram(n_action_2)}")

    p_equal = sum(1 for p_1, p_2 in zip(probabilities_action_1, probabilities_action_2) if p_1 == p_2)
    n_equal = sum(1 for n_1, n_2 in zip(n_action_1, n_action_2) if n_1 == n_2)
    print(f"\nP_action_1 == P_action_2: {p_equal} ({p_equal/len(scenarios):.1%})")
    print(f"N_action_1 == N_action_2: {n_equal} ({n_equal/len(scenarios):.1%})")


# ============================================================================
# ACTION STRUCTURE
# ============================================================================

def report_action_structure(scenarios):
    print("\n=== ACTION STRUCTURE ===")
    n = len(scenarios)

    slot_1_actions = [s["scenario_info"]["slot_1_action"] for s in scenarios]
    slot_2_actions = [s["scenario_info"]["slot_2_action"] for s in scenarios]
    swapped = sum(1 for s in scenarios if s["scenario_info"]["slots_swapped"])

    print(f"\nSlot 1 action distribution: {histogram(slot_1_actions)}")
    print(f"Slot 2 action distribution: {histogram(slot_2_actions)}")
    print(f"Scenarios with slots swapped: {swapped} ({swapped/n:.1%})")

    passengers_action_1 = sum(1 for s in scenarios if s["scenario_info"]["is_passengers_action_1"])
    passengers_action_2 = sum(1 for s in scenarios if s["scenario_info"]["is_passengers_action_2"])
    print(f"\nPassenger scenarios (Action 1): {passengers_action_1} ({passengers_action_1/n:.1%})")
    print(f"Passenger scenarios (Action 2): {passengers_action_2} ({passengers_action_2/n:.1%})")

    lawful_action_1 = sum(1 for s in scenarios if s["scenario_info"]["is_lawful_action_1"])
    lawful_action_2 = sum(1 for s in scenarios if s["scenario_info"]["is_lawful_action_2"])
    print(f"Lawful (Action 1): {lawful_action_1} ({lawful_action_1/n:.1%})")
    print(f"Lawful (Action 2): {lawful_action_2} ({lawful_action_2/n:.1%})")


# ============================================================================
# CHARACTER USAGE
# ============================================================================

def report_character_usage(scenarios):
    print("\n=== CHARACTER USAGE ===")
    all_characters = Counter()
    for s in scenarios:
        for character in s["scenario_info"]["group_action_1"]:
            all_characters[character] += 1
        for character in s["scenario_info"]["group_action_2"]:
            all_characters[character] += 1

    total = sum(all_characters.values())
    print(f"\nTotal character instances: {total}")
    print(f"Distinct character types used: {len(all_characters)}")
    print(f"\nPer-character usage (count, share of all instances):")
    for character, count in sorted(all_characters.items(),
                                   key=lambda kv: (-kv[1], kv[0])):
        print(f"  {character:<18} {count:>5} ({count/total:.1%})")


# ============================================================================
# DERIVED QUANTITIES
# ============================================================================

def report_derived_quantities(scenarios):
    print("\n=== DERIVED QUANTITIES ===")
    n = len(scenarios)

    # Expected harm per side, range [0.0, 5.0] under PRISM-4D defaults
    # (max probability 0.85, max group size 5).
    expected_harm_action_1 = [
        (s["scenario_info"]["probability_action_1"] / 100.0) * s["scenario_info"]["n_action_1"]
        for s in scenarios
    ]
    expected_harm_action_2 = [
        (s["scenario_info"]["probability_action_2"] / 100.0) * s["scenario_info"]["n_action_2"]
        for s in scenarios
    ]
    eh_gaps = [abs(a - b) for a, b in zip(expected_harm_action_1, expected_harm_action_2)]
    print(f"\nExpected harm (Action 1): "
          f"min={min(expected_harm_action_1):.2f}, max={max(expected_harm_action_1):.2f}, "
          f"mean={sum(expected_harm_action_1)/n:.2f}")
    print(f"Expected harm (Action 2): "
          f"min={min(expected_harm_action_2):.2f}, max={max(expected_harm_action_2):.2f}, "
          f"mean={sum(expected_harm_action_2)/n:.2f}")
    print(f"Expected-harm gap |action_1 - action_2|: "
          f"mean={sum(eh_gaps)/n:.2f}, max={max(eh_gaps):.2f}")

    # Equal-expected-harm scenarios (gap < 1e-6, accounts for float drift).
    equal_eh = sum(1 for gap in eh_gaps if gap < 1e-6)
    print(f"Equal-expected-harm scenarios: {equal_eh} ({equal_eh/n:.1%})")

    # Dominance-bearing scenarios: one side has both lower (or equal) P
    # and lower (or equal) N, with at least one strictly lower.
    dominance_bearing = 0
    for s in scenarios:
        info = s["scenario_info"]
        p_1, p_2 = info["probability_action_1"], info["probability_action_2"]
        n_1, n_2 = info["n_action_1"], info["n_action_2"]
        action_1_dominated = (p_2 <= p_1 and n_2 <= n_1 and (p_2 < p_1 or n_2 < n_1))
        action_2_dominated = (p_1 <= p_2 and n_1 <= n_2 and (p_1 < p_2 or n_1 < n_2))
        if action_1_dominated or action_2_dominated:
            dominance_bearing += 1
    print(f"Dominance-bearing scenarios (one side strictly dominates): "
          f"{dominance_bearing} ({dominance_bearing/n:.1%})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze PRISM-4D dataset statistics.")
    parser.add_argument("--scenarios", type=str, default="../data/scenarios.jsonl",
                        help="path to scenarios.jsonl (default: ../data/scenarios.jsonl)")
    args = parser.parse_args()

    with open(args.scenarios) as f:
        scenarios = [json.loads(line) for line in f]

    print(f"=== PRISM-4D dataset statistics ===")
    print(f"Source: {args.scenarios}")
    report_scenario_stats(scenarios)
    report_probability_and_size_stats(scenarios)
    report_action_structure(scenarios)
    report_character_usage(scenarios)
    report_derived_quantities(scenarios)
    print()


if __name__ == "__main__":
    main()