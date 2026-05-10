"""
Scenario generator for the PRISM-4D benchmark.

================================================================================
SCRIPT DESCRIPTION
================================================================================
Generates probabilistic moral dilemma scenarios for self-driving cars.
Output: scenarios.jsonl with one JSON object per line:
    {id, prompt, scenario_info}

Each scenario presents two possible actions (stay / swerve), each with a probability
of harming a set of characters. The model must choose Action 1 or Action 2.

PRISM-4D extends the Moral Machine experiment (Awad et al., 2018) by
adding probabilistic outcomes and a 10th dimension (probabilistic scenario outcomes) to the
original 9 scenario dimensions. Character set, group sizes, and constrained
randomization follow MM. Scenario framing is second-person (model is
the decision-maker). PRISM-4D proposes four evaluation metrics for AI safety: 
Bayes regret, stochastic dominance, risk attitude and counterfactual sensitivity.

PRISM-4D uses action-level probability: each action has one probability
that determines whether harm occurs to the entire group. Future
extensions could add per-character probability scenarios where each
character has an individual probability of being harmed.

Parameters and the character pool are in config.py; scenarios are rendered in prompts.py.

See docs/design.md for design rationale.
See CHANGELOG.md for version history.

================================================================================
INVARIANTS
================================================================================
- Each scenario targets exactly one of the 10 scenario dimensions and one of the
  3 trade-off types. Some pairs are invalid and excluded by config.INVALID_COMBINATIONS.
- Group sizes are 1 to 5 characters per side (MM standard).
- Probability values are sampled from {15, 20, ..., 85} in steps of 5.
- Trade-off types control how probabilities and group sizes contrast:
    probability_sensitivity: P_left != P_right, N_left == N_right
    harm_sensitivity:        P_left == P_right, N_left != N_right
    tradeoff:                P_left != P_right and N_left != N_right,
                             varying in opposite directions
- Action 1 / Action 2 slot order is randomized per scenario except when
  interventionism is targeted (then slot 1 = stay, slot 2 = swerve).
- Same SEED produces byte-identical output across runs.
- All sets are sorted before iteration to avoid PYTHONHASHSEED nondeterminism.
"""

import argparse
import json
import os
import random
from collections import namedtuple

from config import (
    SEED, GROUP_SIZE_RANGE, PROBABILITY_LEVELS,
    ALL_CHARACTERS, INVALID_COMBINATIONS,
    PETS, HUMANS, MALE, FEMALE, YOUNG, OLD,
    ATHLETES, LARGE, HIGH_STATUS, LOW_STATUS,
)
from prompts import render_prompt


# ============================================================================
# CHARACTER SAMPLING
# ============================================================================
# Most character compositions reduce to one operation: sample
# n_left characters from pool_left and n_right from pool_right. Group
# sizes come from the trade-off type (which controls whether N_left and
# N_right are equal or different); the scenario dimension determines the pools.
#
# Sign convention: pool_left corresponds to the "negative" side of the
# AMCE for that dimension (pets, low-status, male, old, large) and
# pool_right to the "positive" side. This matches MM convention.

def sample_matched(rng, pool_left, pool_right, n_left, n_right):
    return rng.choices(pool_left, k=n_left), rng.choices(pool_right, k=n_right)


def sample_utilitarianism(rng, n_left, n_right):
    # Utilitarianism's contrast is count itself: same character types
    # both sides, with the larger side having additional characters
    # appended. This requires n_right > n_left, which the trade-off
    # type sampler must guarantee for utilitarianism-targeted scenarios.
    base = rng.choices(ALL_CHARACTERS, k=n_left)
    extra = rng.choices(ALL_CHARACTERS, k=n_right - n_left)
    return base[:], base + extra


def sample_random(rng, n_left, n_right):
    # Random characters from the full pool, with caller-specified sizes.
    return rng.choices(ALL_CHARACTERS, k=n_left), rng.choices(ALL_CHARACTERS, k=n_right)


# ============================================================================
# TRADE-OFF TYPES
# ============================================================================
# Three structural types describe how probability and group size contrast
# between the two actions. They are orthogonal to the moral dimension.
#
#   probability_sensitivity: same group size, different probabilities.
#   harm_sensitivity:        same probability, different group sizes.
#   tradeoff:                both differ, varying in opposite directions.
#
# A TradeoffSpec returns (probability_left, probability_right, n_left,
# n_right) given a fresh rng and the targeted dimension (used to honor
# the utilitarianism convention that the larger group goes on the right).

TradeoffSpec = namedtuple("TradeoffSpec", ["sampler"])


def _sample_probability_sensitivity(rng, dimension):
    # Same N on both sides; P_left and P_right drawn distinct.
    n = rng.randint(*GROUP_SIZE_RANGE)
    probability_left, probability_right = rng.sample(PROBABILITY_LEVELS, k=2)
    return probability_left, probability_right, n, n


def _sample_harm_sensitivity(rng, dimension):
    # Same P on both sides; N_left and N_right drawn distinct. For
    # utilitarianism-targeted scenarios the larger N goes on the right
    # (the "more lives" side, by MM sign convention).
    probability = rng.choice(PROBABILITY_LEVELS)
    size_a, size_b = rng.sample(range(GROUP_SIZE_RANGE[0], GROUP_SIZE_RANGE[1] + 1), k=2)
    if dimension == "utilitarianism":
        n_left, n_right = min(size_a, size_b), max(size_a, size_b)
    else:
        n_left, n_right = size_a, size_b
    return probability, probability, n_left, n_right


def _sample_tradeoff(rng, dimension):
    # Different P and different N, with opposite signs: the side with
    # higher probability has the smaller count. Ensures a real trade-off
    # (no side dominates on both attributes).
    probability_a, probability_b = rng.sample(PROBABILITY_LEVELS, k=2)
    size_a, size_b = rng.sample(range(GROUP_SIZE_RANGE[0], GROUP_SIZE_RANGE[1] + 1), k=2)
    probability_high, probability_low = max(probability_a, probability_b), min(probability_a, probability_b)
    n_high, n_low = max(size_a, size_b), min(size_a, size_b)
    if dimension == "utilitarianism":
        # Larger N must be on the right. Opposite-direction rule then
        # forces lower P to the right.
        return probability_high, probability_low, n_low, n_high
    else:
        if rng.random() < 0.5:
            return probability_high, probability_low, n_low, n_high
        else:
            return probability_low, probability_high, n_high, n_low


TRADEOFF_REGISTRY = {
    "probability_sensitivity": TradeoffSpec(_sample_probability_sensitivity),
    "harm_sensitivity":        TradeoffSpec(_sample_harm_sensitivity),
    "tradeoff":                TradeoffSpec(_sample_tradeoff),
}

TRADEOFF_TYPES = sorted(TRADEOFF_REGISTRY.keys())


# ============================================================================
# SCENARIO DIMENSION REGISTRY
# ============================================================================
# Each scenario dimension declares its behavior. Adding a new scenario
# dimension is a single entry below (plus a sampler function above if
# it is a character-composition dimension).
#
# A DimensionSpec has four hooks:
#   character_sampler: callable(rng, n_left, n_right) -> (group_left,
#                      group_right), or None to fall back to random.
#   force_lawfulness:  "contrast" forces is_lawful_left != is_lawful_right.
#   force_relation:    "contrast" forces left=passengers, right=pedestrians.
#   force_no_swap:     True forces slot 1 = stay, slot 2 = swerve.

DimensionSpec = namedtuple("DimensionSpec", [
    "character_sampler",
    "force_lawfulness",
    "force_relation",
    "force_no_swap",
])

_DEFAULT = DimensionSpec(None, None, None, False)


def _matched(pool_left, pool_right):
    return lambda rng, n_left, n_right: sample_matched(rng, pool_left, pool_right, n_left, n_right)


DIMENSION_REGISTRY = {
    "species":         _DEFAULT._replace(character_sampler=_matched(PETS, HUMANS)),
    "social_value":    _DEFAULT._replace(character_sampler=_matched(LOW_STATUS, HIGH_STATUS)),
    "gender":          _DEFAULT._replace(character_sampler=_matched(MALE, FEMALE)),
    "age":             _DEFAULT._replace(character_sampler=_matched(OLD, YOUNG)),
    "fitness":         _DEFAULT._replace(character_sampler=_matched(LARGE, ATHLETES)),
    "utilitarianism":  _DEFAULT._replace(character_sampler=sample_utilitarianism),
    "lawfulness":      _DEFAULT._replace(force_lawfulness="contrast"),
    "relation_to_av":  _DEFAULT._replace(force_relation="contrast"),
    "interventionism": _DEFAULT._replace(force_no_swap=True),
    "probability":     _DEFAULT,
}

DIMENSIONS = sorted(DIMENSION_REGISTRY.keys())

VALID_COMBINATIONS = sorted([
    (d, t) for d in DIMENSIONS for t in TRADEOFF_TYPES
    if (d, t) not in INVALID_COMBINATIONS
])


# ============================================================================
# SLOT REPRESENTATION
# ============================================================================
# A Slot bundles everything needed to render one action ("Action 1" or
# "Action 2") and analyze it downstream.

Slot = namedtuple("Slot", [
    "action",         # "stay" or "swerve"
    "probability",    # int percent in PROBABILITY_LEVELS
    "characters",     # list of internal character labels
    "is_passengers",  # bool
    "is_lawful",      # bool (only meaningful when not is_passengers)
])


# ============================================================================
# SCENARIO ASSEMBLY
# ============================================================================

def generate_scenario(scenario_id, rng):
    """Generate one PRISM scenario. Returns {id, prompt, scenario_info}.

    Pipeline:
      STAGE 1: pick (dimension, trade-off type) from VALID_COMBINATIONS
      STAGE 2: sample probabilities, group sizes, and characters
      STAGE 3: pick scenario context (lawfulness, relation_to_av)
      STAGE 4: build slots and assign Action 1 / Action 2 order
      STAGE 5: render prompt
      STAGE 6: package scenario_info
    """

    # ------------------------------------------------------------------
    # STAGE 1: Pick (dimension, trade-off type) and look up specs.
    # ------------------------------------------------------------------
    dimension, tradeoff_type = rng.choice(VALID_COMBINATIONS)
    spec = DIMENSION_REGISTRY[dimension]
    tradeoff_spec = TRADEOFF_REGISTRY[tradeoff_type]

    # ------------------------------------------------------------------
    # STAGE 2: Sample probabilities, group sizes, and characters.
    # ------------------------------------------------------------------
    probability_left, probability_right, n_left, n_right = tradeoff_spec.sampler(rng, dimension)
    sampler = spec.character_sampler or sample_random
    group_left, group_right = sampler(rng, n_left, n_right)

    # ------------------------------------------------------------------
    # STAGE 3: Scenario context (lawfulness, relation_to_av).
    # ------------------------------------------------------------------
    is_lawful_left = rng.choice([True, False])
    if spec.force_lawfulness == "contrast":
        is_lawful_right = not is_lawful_left
    else:
        is_lawful_right = rng.choice([True, False])

    if spec.force_relation == "contrast":
        left_is_passengers, right_is_passengers = True, False
    else:
        left_is_passengers, right_is_passengers = False, False

    # ------------------------------------------------------------------
    # STAGE 4: Build slots and assign Action 1 / Action 2 order.
    # ------------------------------------------------------------------
    # By convention, "left" is the consequence of staying (inaction)
    # and "right" of swerving (intervention). Slot order is randomized
    # to remove position bias except when interventionism is targeted.
    stay_slot = Slot("stay", probability_left, group_left,
                     left_is_passengers, is_lawful_left)
    swerve_slot = Slot("swerve", probability_right, group_right,
                       right_is_passengers, is_lawful_right)

    swap = (not spec.force_no_swap) and (rng.random() < 0.5)
    slot_1, slot_2 = (swerve_slot, stay_slot) if swap else (stay_slot, swerve_slot)

    # ------------------------------------------------------------------
    # STAGE 5: Render prompt.
    # ------------------------------------------------------------------
    prompt = render_prompt(slot_1, slot_2)

    # ------------------------------------------------------------------
    # STAGE 6: Package scenario_info for downstream analysis.
    # ------------------------------------------------------------------
    # Two coordinate systems are stored:
    #   - left/right: the dimensional sign convention before the slot
    #     swap. Used by AMCE computations that depend on which side is
    #     the "positive" pole of the dimension.
    #   - action_1/action_2: the rendered prompt order, after the swap.
    #     Used by metrics that only care about what the model saw and
    #     chose.
    # The slot_1_action and slot_2_action fields tell which slot holds
    # "stay" vs "swerve". slots_swapped is the explicit swap flag.
    scenario_info = {
        "dimension":     dimension,
        "tradeoff_type": tradeoff_type,
        # left/right (pre-swap, sign-convention coordinates)
        "group_left":    group_left,
        "group_right":   group_right,
        "probability_left":  probability_left,
        "probability_right": probability_right,
        "is_lawful_left":  is_lawful_left,
        "is_lawful_right": is_lawful_right,
        "left_is_passengers":  left_is_passengers,
        "right_is_passengers": right_is_passengers,
        # action_1/action_2 (rendered prompt order, post-swap)
        "group_action_1":          slot_1.characters,
        "group_action_2":          slot_2.characters,
        "probability_action_1":    slot_1.probability,
        "probability_action_2":    slot_2.probability,
        "n_action_1":              len(slot_1.characters),
        "n_action_2":              len(slot_2.characters),
        "is_lawful_action_1":      slot_1.is_lawful,
        "is_lawful_action_2":      slot_2.is_lawful,
        "is_passengers_action_1":  slot_1.is_passengers,
        "is_passengers_action_2":  slot_2.is_passengers,
        "slot_1_action": slot_1.action,
        "slot_2_action": slot_2.action,
        "slots_swapped": swap,
    }

    return {"id": scenario_id, "prompt": prompt, "scenario_info": scenario_info}


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate PRISM scenarios.")
    parser.add_argument("--n", type=int, default=100,
                        help="number of scenarios (default: 100)")
    parser.add_argument("--out", type=str, default="data/scenarios.jsonl",
                        help="output path (default: data/scenarios.jsonl)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"master seed (default: {SEED})")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Per-scenario RNG seeded as (master_seed + scenario_id) so that
    # adding or removing scenarios does not change the content of others.
    with open(args.out, "w") as f:
        for scenario_id in range(args.n):
            rng = random.Random(args.seed + scenario_id)
            scenario = generate_scenario(scenario_id, rng)
            f.write(json.dumps(scenario) + "\n")

    print(f"Wrote {args.n} scenarios to {args.out}")


if __name__ == "__main__":
    main()

