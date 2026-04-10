# COMP0223 Self-Touch Detailed Deck

This is the `more detailed` version of the coursework presentation.

It is still framed as:

`inspired by infant self-touch developmental studies, implemented in BabyBench with intrinsic-motivation reinforcement learning`

It intentionally does **not** mention `Baby Sophia`.

## Main Motivation Paper

`Thomas, B. L., Karl, J. M., & Whishaw, I. Q. (2015). Independent development of the reach and the grasp in spontaneous self-touching by human infants in the first 6 months. Frontiers in Psychology, 5, 1526.`

## Supporting Motivation References

- `Rochat, P. (1998). Self-perception and action in infancy. Experimental Brain Research.`
- `DiMercurio, A., Connell, J. P., Clark, M., & Corbetta, D. (2018). A naturalistic observation of spontaneous touches to the body and environment in the first 2 months of life. Frontiers in Psychology.`

## Slide 1

**Title**

`Learning Infant-Like Self-Touch in BabyBench`

**Purpose**

Introduce the task, benchmark, and research question in one clear slide.

**Visual**

- `presentation/assets_detailed/slide1_cover_detailed.png`

**Key message**

The project studies whether self-touch behaviour can emerge from intrinsic rewards alone.

---

## Slide 2

**Title**

`Motivation 1: what infant self-touch studies suggest`

**Purpose**

Explain the developmental psychology motivation in more detail than the concise deck.

**Visual**

- `presentation/assets_detailed/slide2_motivation_detailed.png`

**Key points**

- Infant self-touch appears very early.
- It is self-generated rather than externally instructed.
- It naturally produces both tactile and proprioceptive signals.
- Repeated self-touch contributes to body awareness and later sensorimotor development.

---

## Slide 3

**Title**

`Motivation 2: why this is a meaningful robotics problem`

**Purpose**

Bridge the infant study literature to the robotics learning problem.

**Visual**

- `presentation/assets_detailed/slide3_robotics_motivation.png`

**Key points**

- Self-touch is a useful example of embodied autonomous learning.
- It provides a naturally multimodal signal without labels.
- BabyBench is a good platform because it offers infant-like embodiment, continuous control, and benchmarked evaluation.

---

## Slide 4

**Title**

`Method: intrinsic-motivation RL pipeline`

**Purpose**

Explain the implementation clearly enough that the audience understands what you actually built.

**Visual**

- `presentation/assets_detailed/slide4_method_detailed.png`

**Key points**

- Touch and proprioception are the main observations.
- Touch is compressed into 68 grouped features.
- PPO is used to optimize the policy.
- Training uses a three-stage curriculum.
- Final model selection is based on standard evaluation performance.

---

## Slide 5

**Title**

`Detailed algorithmic design`

**Purpose**

Explain the per-step learning loop and show clearly how observation processing, intrinsic rewards, and PPO fit together.

**Visual**

- `presentation/assets_detailed/slide5_algorithm_detailed.png`

**Key points**

- Touch is flattened and compressed into 68 grouped tactile features.
- The intrinsic reward is computed in a wrapper from novelty, contact, milestones, and balance.
- PPO interacts with the wrapped environment through a standard rollout / update loop.
- Final model selection is done by deterministic evaluation, not by training coverage alone.

---

## Slide 6

**Title**

`Training progress across the three stages`

**Purpose**

Use plots to explain the difference between raw exploration success during training and final model choice.

**Visual**

- `presentation/assets_detailed/slide6_training_detailed.png`

**Key points**

- Base stage: stable but limited coverage.
- Difficult stage: strongest exploration under randomized initial states.
- After stage: best checkpoint under deterministic evaluation.

---

## Slide 7

**Title**

`Final evaluation, interpretation, and takeaway`

**Purpose**

Summarize the final result honestly and clearly.

**Visual**

- `presentation/assets_detailed/slide7_results_detailed.png`

**Key points**

- Final chosen model: after-stage final checkpoint.
- Deterministic evaluation score: `0.2206`.
- After-stage outperformed difficult-stage under standard evaluation.
- The result is meaningful, but still limited in generalization and body coverage.

## Suggested Timing

- Slide 1: `20s`
- Slide 2: `35s`
- Slide 3: `30s`
- Slide 4: `35s`
- Slide 5: `30s`
- Slide 6: `30s`
- Slide 7: `25s`

Approximate total:

- `about 3 minutes 20 seconds to 3 minutes 35 seconds`

If you need to compress to a strict `3:00`, shorten Slide 3 and Slide 4 first.
