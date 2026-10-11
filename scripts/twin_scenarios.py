"""
Twin scenarios for the PRISM-4D benchmark.

This script implements the "future extension" named in evaluation_metrics.py
for metric 4 (counterfactual sensitivity): "true matched-pair counterfactual
sensitivity by generating scenario pairs that differ on exactly one
attribute".
generate.py, config.py and prompts.py are not changed.

================================================================================
WHAT A TWIN IS
================================================================================
For every scenario whose targeted dimension is one of the five character-
composition dimensions (species, social_value, gender, age, fitness), the
twin is the same scenario with only the targeted attribute changed:

  - every character is replaced by its counterpart on the other side of the
    dimension (e.g. gender: male doctor -> female doctor; fitness: large
    man -> male athlete), so that the other attributes of the character stay
    the same where the character pool allows it;
  - where the pool has no one-to-one counterpart (e.g. species: a human has
    no matching pet), the replacement is drawn at random from the opposite
    sub-pool of config.py, with a seeded random generator;
  - probabilities, group sizes, actions (stay / swerve), context (lawfulness,
    passengers / pedestrians) and the Action 1 / Action 2 order are kept.

In the original scenarios the positive pole (humans, female, young, athletic,
high status) is always on the swerve side (generate.py: "right" = swerve).
In the twin it is on the stay side, so the pair also separates the attribute
from the stay / swerve action.

For lawfulness scenarios, the twin swaps the traffic signals of the two groups
(green <-> red) and keeps everything else, the characters included. In the
original the lawful group is on the stay or the swerve side at random; in the
twin it is on the other side.

utilitarianism is not twinned: its attribute is the group size itself, which
is part of expected harm. interventionism, relation_to_av and probability are
not twinned in this version.

================================================================================
USAGE
================================================================================
    python scripts/twin_scenarios.py \
        --scenarios data/scenarios_1000.jsonl --out data/scenarios_twins.jsonl

For the 1,000 scenarios this writes 645 twins. --dimensions restricts the
twins to some of the six dimensions.

Each line of the output has the format of scenarios.jsonl, with id =
TWIN_ID_OFFSET + id of the original and two extra scenario_info fields:
"twin_of" (id of the original) and "positive_pole_side" (side of the positive
pole in the twin: "left" = stay, "right" = swerve; for the composition
dimensions always "left", for lawfulness the side of the lawful group). The
same input gives byte-identical output.
"""

import argparse
import json
import random

from config import CHARACTERS, PETS, HUMANS, MALE, FEMALE, YOUNG, OLD, ATHLETES, LARGE, HIGH_STATUS, LOW_STATUS
from generate import Slot
from prompts import render_prompt

COMPOSITION_DIMENSIONS = ["species", "social_value", "gender", "age", "fitness"]
TWIN_DIMENSIONS = COMPOSITION_DIMENSIONS + ["lawfulness"]
TWIN_ID_OFFSET = 10000      # twin of scenario k has id 10000 + k
TWIN_SEED_OFFSET = 10000    # random replacements use random.Random(10000 + k)

# The two sides of each dimension, as in generate.DIMENSION_REGISTRY:
# (negative pool = "left" in the originals, positive pool = "right").
POOLS = {
    "species":      (PETS, HUMANS),
    "social_value": (LOW_STATUS, HIGH_STATUS),
    "gender":       (MALE, FEMALE),
    "age":          (OLD, YOUNG),
    "fitness":      (LARGE, ATHLETES),
}

# One-to-one counterparts where the pool has them. Characters missing here are
# replaced at random from the opposite pool.
COUNTERPARTS = {
    "gender": {
        "man": "woman", "boy": "girl", "old_man": "old_woman", "large_man": "large_woman",
        "executive_male": "executive_female", "athlete_male": "athlete_female",
        "doctor_male": "doctor_female",
        "woman": "man", "girl": "boy", "old_woman": "old_man", "large_woman": "large_man",
        "executive_female": "executive_male", "athlete_female": "athlete_male",
        "doctor_female": "doctor_male",
        # pregnant_woman has no male counterpart: replaced at random from MALE.
    },
    "age": {
        "old_man": "boy", "old_woman": "girl", "boy": "old_man", "girl": "old_woman",
        # stroller (baby in stroller) has no old counterpart: replaced at random from OLD.
    },
    "fitness": {
        "large_man": "athlete_male", "large_woman": "athlete_female",
        "athlete_male": "large_man", "athlete_female": "large_woman",
    },
    # species and social_value: no one-to-one counterparts in the pool.
    "species": {},
    "social_value": {},
}


def counterpart(character, dimension, random_generator):
    """Character on the other side of the dimension."""
    if character in COUNTERPARTS[dimension]:
        return COUNTERPARTS[dimension][character]
    negative_pool, positive_pool = POOLS[dimension]
    opposite_pool = positive_pool if character in negative_pool else negative_pool
    assert character in negative_pool or character in positive_pool, (character, dimension)
    return random_generator.choice(opposite_pool)


def slots_from_scenario_info(scenario_info, group_left, group_right):
    """Rebuild the two slots exactly as generate.generate_scenario does (stage 4)."""
    stay_slot = Slot("stay", scenario_info["probability_left"], group_left,
                     scenario_info["left_is_passengers"], scenario_info["is_lawful_left"])
    swerve_slot = Slot("swerve", scenario_info["probability_right"], group_right,
                       scenario_info["right_is_passengers"], scenario_info["is_lawful_right"])
    return (swerve_slot, stay_slot) if scenario_info["slots_swapped"] else (stay_slot, swerve_slot)


def make_lawfulness_twin(scenario):
    """Same scenario with the traffic signals of the two groups swapped."""
    scenario_info = scenario["scenario_info"]
    assert not scenario_info["left_is_passengers"] and not scenario_info["right_is_passengers"]   # both groups are pedestrians
    assert scenario_info["is_lawful_left"] != scenario_info["is_lawful_right"]                    # one lawful, one not
    twin_info = dict(scenario_info)
    twin_info.update({"is_lawful_left": scenario_info["is_lawful_right"], "is_lawful_right": scenario_info["is_lawful_left"]})
    slot_1, slot_2 = slots_from_scenario_info(twin_info, scenario_info["group_left"], scenario_info["group_right"])
    twin_info.update({
        "is_lawful_action_1": slot_1.is_lawful,
        "is_lawful_action_2": slot_2.is_lawful,
        "twin_of": scenario["id"],
        "positive_pole_side": "left" if twin_info["is_lawful_left"] else "right",
    })
    # Characters, probabilities, group sizes, actions and order are unchanged.
    for key in ("group_action_1", "group_action_2", "probability_action_1", "probability_action_2",
                "n_action_1", "n_action_2", "slot_1_action", "slot_2_action", "slots_swapped"):
        assert twin_info[key] == scenario_info[key]
    assert twin_info["is_lawful_action_1"] != scenario_info["is_lawful_action_1"]
    return {"id": TWIN_ID_OFFSET + scenario["id"], "prompt": render_prompt(slot_1, slot_2),
            "scenario_info": twin_info}


def make_twin(scenario):
    scenario_info = scenario["scenario_info"]
    dimension = scenario_info["dimension"]
    if dimension == "lawfulness":
        return make_lawfulness_twin(scenario)
    random_generator = random.Random(TWIN_SEED_OFFSET + scenario["id"])
    twin_group_left = [counterpart(character, dimension, random_generator) for character in scenario_info["group_left"]]
    twin_group_right = [counterpart(character, dimension, random_generator) for character in scenario_info["group_right"]]
    negative_pool, positive_pool = POOLS[dimension]
    assert all(character in positive_pool for character in twin_group_left) and all(character in negative_pool for character in twin_group_right)
    slot_1, slot_2 = slots_from_scenario_info(scenario_info, twin_group_left, twin_group_right)
    twin_info = dict(scenario_info)
    twin_info.update({
        "group_left": twin_group_left,
        "group_right": twin_group_right,
        "group_action_1": slot_1.characters,
        "group_action_2": slot_2.characters,
        "twin_of": scenario["id"],
        "positive_pole_side": "left",
    })
    # Everything else (probabilities, group sizes, actions, context, order) is unchanged.
    for key in ("probability_action_1", "probability_action_2", "n_action_1", "n_action_2",
                "slot_1_action", "slot_2_action", "slots_swapped"):
        assert twin_info[key] == scenario_info[key]
    assert len(slot_1.characters) == scenario_info["n_action_1"] and len(slot_2.characters) == scenario_info["n_action_2"]
    return {"id": TWIN_ID_OFFSET + scenario["id"], "prompt": render_prompt(slot_1, slot_2),
            "scenario_info": twin_info}


def main():
    parser = argparse.ArgumentParser(description="Generate twin scenarios of PRISM-4D scenarios.")
    parser.add_argument("--scenarios", type=str, default="data/scenarios_1000.jsonl")
    parser.add_argument("--out", type=str, default="data/scenarios_twins.jsonl")
    parser.add_argument("--dimensions", nargs="+", default=TWIN_DIMENSIONS, choices=TWIN_DIMENSIONS,
                        help="dimensions to twin (default: all six)")
    arguments = parser.parse_args()
    scenarios = [json.loads(line) for line in open(arguments.scenarios)]
    twins = []
    for scenario in scenarios:
        scenario_info = scenario["scenario_info"]
        # Check: rebuilding the original from its stored variables gives its prompt exactly.
        assert render_prompt(*slots_from_scenario_info(scenario_info, scenario_info["group_left"], scenario_info["group_right"])) == scenario["prompt"]
        if scenario_info["dimension"] in arguments.dimensions:
            twins.append(make_twin(scenario))
    with open(arguments.out, "w") as twin_file:
        for twin in twins:
            twin_file.write(json.dumps(twin) + "\n")
    print(f"Wrote {len(twins)} twins of {len(scenarios)} scenarios to {arguments.out}")


if __name__ == "__main__":
    main()
