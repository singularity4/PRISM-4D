# PRISM-4D Design 

PRISM-4D extends the Moral Machine experiment (Awad et al., 2018) to text
scenarios with probabilistic outcomes and four evaluation metrics. 

## Scope

PRISM-4D measures how language models reason about moral dilemmas in
self-driving car scenarios. It is not a policy prescription for smart car manufacturers.
Like the original Moral Machine, this research is descriptive — its
purpose is to explain what models do, not what they should do.

PRISM-4D's contribution over the original Moral Machine and prior LLM
adaptations (Takemoto 2024, MoralBench) is threefold: probabilistic
outcomes (addressing Schuessler 2023's critique of the original MM's
certain-outcome assumption), value-aligned chain-of-thought (VA-CoT)
as a reasoning scaffold, and a 4D evaluation framework relevant for AI safety.
This document covers scenario generation; VA-CoT and the evaluation are separate.

## Scenario design

Each scenario presents two actions — stay or swerve — and asks the
model to choose only one. Scenarios are framed in the second person: the model
is the decision-making system of the car, not an outside observer.

Outcomes are probabilistic. Each action has a stated probability of
hitting a set of characters; with the complementary probability, no
harm occurs. The original Moral Machine forces certain outcomes ("3
will die"). PRISM-4D's "P% chance of hitting" formulation matches how risk
actually presents in deployment and lets risk attitude be measured.

## Scenario dimensions

Ten scenario dimensions vary across scenarios. Nine are inherited from the
original Moral Machine; the tenth is PRISM-4D's probabilistic extension.

The six primary dimensions vary character composition: species
(humans vs. pets), social value (high-status vs. low-status), gender,
age (young vs. old), fitness (athletes vs. large persons), and
utilitarianism (more vs. fewer characters).

The three secondary dimensions vary scenario context: lawfulness
(crossing on green vs. red), interventionism (stay vs. swerve), and
relation to AV (passengers vs. pedestrians).

The PRISM-4D new scenario dimension is probability. Probability values appear in every
scenario regardless of the targeted dimension; when probability is
itself the targeted dimension, the trade-off type controls how the two
sides contrast (see Trade-off types below).

Each scenario targets exactly one dimension via constrained
randomization, following Awad et al. The targeted dimension's contrast
is enforced; the other nine vary freely. This is what makes the average
marginal component effect (AMCE) identifiable per dimension.

## Probability range

Probabilities are sampled from {15, 20, 25, ..., 85} in steps of 5.

The lower and upper bounds exclude degenerate cases: 0% means no
dilemma, 100% reduces to the original MM's certain-outcome formulation
that PRISM-4D explicitly avoids. Values very close to 0 or 100 also feel
artificial in scenario text.

Steps of 5 give 15 levels — fine-grained enough to detect risk
sensitivity, coarse enough that AMCE can be estimated with a manageable
number of scenarios. Round numbers are easier for models to reason
about than fractional values.

## Trade-off types

Every scenario combines a moral dimension with one of three trade-off
types. The trade-off type determines how probability and group size
contrast between the two actions:

- **Probability-sensitivity**: same group size on both sides, different
  probabilities. Tests how the model responds to probability alone,
  holding harm size constant.
- **Harm-sensitivity**: same probability on both sides, different group
  sizes. Tests how the model responds to harm size alone, holding
  probability constant.
- **Trade-off**: probability and group size both differ, varying in
  opposite directions (the side with higher probability has the smaller
  group). Tests how the model resolves a genuine trade-off between
  probability and harm size.

These types are orthogonal to the moral dimension axis: any of the 10
dimensions can in principle be paired with any of the 3 trade-off
types. Two combinations are excluded as logically inconsistent:

- (probability, harm-sensitivity): probability is the targeted contrast,
  but harm-sensitivity forces equal probabilities — contradiction.
- (utilitarianism, probability-sensitivity): utilitarianism contrasts
  group sizes, but probability-sensitivity forces equal sizes —
  contradiction.

This leaves 28 valid (dimension, trade-off-type) combinations, sampled
uniformly per scenario.

PRISM-4D deliberately does not pre-combine probability and group size into
a single expected-value metric (e.g., probability x count). Probability
is a property of the action; group size is a property of the outcome.
The two are kept as separate fields in scenario_info so that downstream
analysis can apply whatever combination logic the question needs,
including expected harm for utilitarian comparison.

## Future extensions

PRISM-4D uses action-level probability: a single probability per action
determines whether the entire group is harmed. One possible extension is
per-character probability, where each character within an action carries
its own probability of being harmed (e.g., "a dog has a 100% chance of
being hit, and a person has a 50% chance"). The two formats test
different aspects of risk reasoning:

- **Action-level (current)** matches MM's original framing: outcomes
  are group-level events, everyone in a group shares one fate.
- **Per-character (future)** models harm as individual independent
  events, closer to real-world physical risk where some people in a
  group may be hit and others spared.

We will also add a random choice as a baseline check (Nagler & Helbing).


## Character pool

PRISM-4D uses the 20 characters from the original Moral Machine without
modification. These are: man, woman, pregnant woman, baby in stroller,
boy, girl, old man, old woman, homeless person, large man, large woman,
criminal, male and female executives, male and female athletes, male
and female doctors, dog, cat.

The character pool encodes implicit value judgments in the original MM
(criminals as low social value, doctors as high). PRISM-4D inherits these
because (a) it preserves direct comparability with prior work and (b)
PRISM-4D measures models, not humans — the value tags are a way to detect
whether models exhibit the same implicit hierarchies, not an
endorsement of those hierarchies.

`pregnant_woman` appears in both the FEMALE and HIGH_STATUS sub-pools.
This is intentional: she is a valid sample under either contrast.

Group sizes range from 1 to 5 characters per side, matching MM.

## Slot order

Action 1 / Action 2 slot order is randomized per scenario to remove
position bias — except when interventionism is the targeted dimension,
in which case slot 1 is fixed to stay and slot 2 to swerve so the
contrast is consistent across scenarios. The stored `slots_swapped`
flag preserves this for downstream analysis.

## Reproducibility

The generator is deterministic: the same SEED produces byte-identical
output across runs. Each scenario's RNG is seeded as
(master_seed + scenario_id), so adding or removing scenarios from a
run does not change the content of other scenarios. All sub-pools
are sorted lists (not sets) to avoid PYTHONHASHSEED nondeterminism.

## Extending the benchmark

Adding a new character-composition dimension requires:

1. Defining a sub-pool list near the existing pools in `generate.py`.
2. Adding one entry to `DIMENSION_REGISTRY` using the `_matched()`
   factory.

Adding a non-character dimension (e.g., a new context flag) requires:

1. Adding a field to `DimensionSpec`.
2. Updating `_DEFAULT` accordingly.
3. Handling the new field in stage 3 of `generate_scenario`.
4. Adding the dimension to `DIMENSION_REGISTRY`.

Adding a new trade-off type requires:

1. Writing a sampler function returning (p_left, p_right, n_left, n_right).
2. Adding an entry to `TRADEOFF_REGISTRY`.
3. Optionally extending `INVALID_COMBINATIONS` if some dimensions
   should not pair with the new type.

A sanity assertion at module load verifies that `DIMENSIONS` and
`DIMENSION_REGISTRY` stay in sync.

## Limitations

Some limitations of the MM such as binary-choice format, the stylized character pool, 
the absence of a neutral option (Bigman & Gray 2020) — are inherited unchanged.
Critiques of the MM that target its use as a policy spec do not apply
to PRISM-4D, which is a quantitative measurement of model behavior. Critiques targeting
its measurement structure (Schuessler 2023, LaCroix 2022) informed the PRISM-4D evaluation framework. 

## References

Awad, E., Dsouza, S., Kim, R., Schulz, J., Henrich, J., Shariff, A.,
Bonnefon, J.-F., & Rahwan, I. (2018). The Moral Machine experiment.
*Nature*, 563, 59–64.

Schuessler, D. (2023). The probability problems of the Moral Machine
experiment. *AI Ethics*.

Takemoto, K. (2024). The Moral Machine experiment on large language
models. *Royal Society Open Science*, 11, 231393.

Bigman, Y. E., & Gray, K. (2020). Life and death decisions of
autonomous vehicles. *Nature*, 579, E1–E2.

LaCroix, T. (2022). Moral dilemmas for moral machines. *AI Ethics*, 2,
737–746.
