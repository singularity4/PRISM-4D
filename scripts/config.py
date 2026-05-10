"""
Configuration for the PRISM-4D benchmark.

Implements all design parameters, the character pool, and invalid combination
rules. 
"""


# ============================================================================
# SCENARIO DESIGN PARAMETERS
# ============================================================================

SEED = 0
GROUP_SIZE_RANGE = (1, 5)                       # min, max characters per side
PROBABILITY_LEVELS = list(range(15, 86, 5))     # {15, 20, ..., 85}


# ============================================================================
# CHARACTER POOL (20 characters, from Moral Machine)
# ============================================================================
# Each entry: internal_label -> (singular_text, plural_text).
# Internal labels are stable identifiers stored in scenario_info; text
# forms are what the model sees in the rendered prompt.

CHARACTERS = {
    "man":              ("man",                "men"),
    "woman":            ("woman",              "women"),
    "pregnant_woman":   ("pregnant woman",     "pregnant women"),
    "stroller":         ("baby in stroller",   "babies in strollers"),
    "boy":              ("boy",                "boys"),
    "girl":             ("girl",               "girls"),
    "old_man":          ("old man",            "old men"),
    "old_woman":        ("old woman",          "old women"),
    "homeless":         ("homeless person",    "homeless persons"),
    "large_man":        ("large man",          "large men"),
    "large_woman":      ("large woman",        "large women"),
    "criminal":         ("criminal",           "criminals"),
    "executive_male":   ("male executive",     "male executives"),
    "executive_female": ("female executive",   "female executives"),
    "athlete_male":     ("male athlete",       "male athletes"),
    "athlete_female":   ("female athlete",     "female athletes"),
    "doctor_male":      ("male doctor",        "male doctors"),
    "doctor_female":    ("female doctor",      "female doctors"),
    "dog":              ("dog",                "dogs"),
    "cat":              ("cat",                "cats"),
}

ALL_CHARACTERS = sorted(CHARACTERS.keys())

# Sub-pools for dimension-targeted sampling. Each is a sorted list (not
# a set) for deterministic iteration under varying PYTHONHASHSEED.
# Note: pregnant_woman intentionally appears in both FEMALE and
# HIGH_STATUS — she is a valid sample under either contrast.

PETS        = sorted(["dog", "cat"])
HUMANS      = sorted([c for c in CHARACTERS if c not in PETS])
MALE        = sorted(["man", "boy", "old_man", "large_man",
                      "executive_male", "athlete_male", "doctor_male"])
FEMALE      = sorted(["woman", "pregnant_woman", "girl", "old_woman", "large_woman",
                      "executive_female", "athlete_female", "doctor_female"])
YOUNG       = sorted(["stroller", "boy", "girl"])
OLD         = sorted(["old_man", "old_woman"])
ATHLETES    = sorted(["athlete_male", "athlete_female"])
LARGE       = sorted(["large_man", "large_woman"])
HIGH_STATUS = sorted(["executive_male", "executive_female",
                      "doctor_male", "doctor_female", "pregnant_woman"])
LOW_STATUS  = sorted(["homeless", "criminal"])


# ============================================================================
# INVALID (DIMENSION, TRADE-OFF) COMBINATIONS
# ============================================================================
# Some pairings contradict their own definitions:
#   probability + harm_sensitivity:
#     probability is the targeted contrast, but harm_sensitivity forces
#     P_left == P_right.
#   utilitarianism + probability_sensitivity:
#     utilitarianism contrasts group size, but probability_sensitivity
#     forces N_left == N_right.

INVALID_COMBINATIONS = {
    ("probability",    "harm_sensitivity"),
    ("utilitarianism", "probability_sensitivity"),
}

