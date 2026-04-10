# COMP0223 Self-Touch Video Deck

This deck is written for a `3-minute coursework video` and is intentionally framed as:

`inspired by infant self-touch developmental studies, implemented in BabyBench with intrinsic-motivation reinforcement learning`

It does **not** mention `Baby Sophia`.

## Main Reference

`Thomas, B. L., Karl, J. M., & Whishaw, I. Q. (2015). Independent development of the reach and the grasp in spontaneous self-touching by human infants in the first 6 months. Frontiers in Psychology, 5, 1526.`

## Slide 1

**Title**

`Learning Infant-Like Self-Touch in BabyBench`

**On-slide text**

- `Task: self-touch learning without extrinsic rewards`
- `Platform: BabyBench + MIMo`
- `Approach: intrinsic-motivation RL`

**Visual**

- Main image: `presentation/assets/slide1_hero_frame.png`

**Speaker script**

```text
Hello, in this project I study how an infant-like robot can learn self-touch autonomously in simulation.

I use the BabyBench platform and the MIMo infant model, and I train the policy with reinforcement learning using intrinsic rewards only. The main question is whether the robot can gradually discover its own body through self-generated exploration, without any external supervision.
```

**Target time**

- `20 to 25 seconds`

---

## Slide 2

**Title**

`Motivation from Infant Self-Touch Development`

**On-slide text**

- `Infants spontaneously touch their own bodies very early in life`
- `Self-touch supports body awareness and sensorimotor development`
- `This developmental idea motivates autonomous body exploration in robots`

**Reference line**

- `Thomas et al., Frontiers in Psychology, 2015`
- Optional small text: `Rochat, Experimental Brain Research, 1998`

**Visual**

- Main image: `presentation/assets/slide2_concept_graphic.png`

**Speaker script**

```text
The motivation for this project comes from developmental studies of infant self-touch.

Thomas and colleagues showed that spontaneous self-touch appears very early in infancy and contributes to the development of body awareness and motor coordination. In other words, by moving and touching their own bodies, infants build a sensorimotor understanding of themselves.

This idea motivates my project: instead of giving the robot an external task reward, I encourage it to explore its own body through intrinsic motivation.
```

**Target time**

- `30 to 35 seconds`

---

## Slide 3

**Title**

`Method: Intrinsic-Motivation RL in BabyBench`

**On-slide text**

- `Observations: touch + proprioception`
- `Touch compressed into 68 grouped features`
- `Intrinsic rewards for novelty, contact coverage, milestones, and balance`
- `PPO training with a 3-stage curriculum: base -> difficult -> after`

**Visual**

- Main image: `presentation/assets/slide3_pipeline_graphic.png`

**Speaker script**

```text
My method uses tactile and proprioceptive observations as the main sensory inputs.

Because the raw touch signal is very high-dimensional, I compress it into 68 grouped touch features and combine these with proprioceptive features inside the policy network. The reward is intrinsic and encourages novelty, new hand-body contacts, broader body coverage, milestone progress, and balanced exploration.

For training, I use PPO and a three-stage curriculum: a base stage, a more difficult exploration stage with randomized initial joint states, and a final after-stage for refinement under a more standard setup.
```

**Target time**

- `35 to 40 seconds`

---

## Slide 4

**Title**

`Training Progress Across Three Stages`

**On-slide text**

- `Base: 5.0M steps, stable but limited coverage`
- `Difficult: 2.0M steps, strongest exploration under curriculum`
- `After: 2.0M steps, best checkpoint under standard evaluation`

**Recommended visual**

- Use this composite figure directly:
  - `presentation/assets/slide4_training_progress.png`

**Supporting numbers**

- `Base: return -13.13, left/right geoms 10 / 11`
- `Difficult: return 2.07, left/right geoms 32 / 34`
- `After: return 2.65, left/right geoms 18 / 10`

**Speaker script**

```text
This slide shows the training dynamics across the three stages.

In the base stage, the policy learned stable self-contact behaviour, but the overall body coverage remained limited. In the difficult stage, the curriculum with randomized initial states greatly increased exploration and contact coverage. Finally, in the after-stage, the policy became more stable again under standard training conditions.

So the main training takeaway is that curriculum improved exploration, while the final stage was important for selecting a model that works better under standard evaluation.
```

**Target time**

- `40 to 45 seconds`

---

## Slide 5

**Title**

`Final Evaluation and Takeaway`

**On-slide text**

- `Final selected model: after-stage final checkpoint`
- `10-episode deterministic evaluation`
- `Manual self-touch score: 0.2206`
- `After-stage outperformed difficult-stage under standard evaluation`

**Visual**

- Preferred static figure: `presentation/assets/slide5_final_results.png`
- Preferred video clip: `results_author_replica/intrinsic_motivation_stelios_x_giannis_after_difficult_task/videos/evaluation_0.avi`
- Alternate still image: `presentation/assets/slide5_demo_frame_alt.png`

**Speaker script**

```text
For the final result, I selected the after-stage model because it performed better under standard deterministic evaluation.

In a 10-episode evaluation, the final policy achieved a cumulative manual self-touch score of about 0.22. The behaviour is still limited and far from full body coverage, but it clearly demonstrates autonomous self-exploration emerging from intrinsic motivation alone.

Overall, this project supports the developmental robotics idea that body knowledge can emerge through self-generated interaction, while also showing that better representation learning and stronger generalization remain important future directions.
```

**Target time**

- `35 to 40 seconds`

---

## Timing Summary

- `Slide 1: 20 to 25 seconds`
- `Slide 2: 30 to 35 seconds`
- `Slide 3: 35 to 40 seconds`
- `Slide 4: 40 to 45 seconds`
- `Slide 5: 35 to 40 seconds`

Target total:

- `about 175 to 185 seconds`

## Delivery Notes

- Use the phrase `inspired by infant developmental studies`.
- Use the phrase `implemented in BabyBench`.
- Use the phrase `final checkpoint selected by evaluation performance`.
- Do **not** say `replicated Baby Sophia`.
- Do **not** say `copied from a paper`.

## Generated Assets

After running `presentation/build_selftouch_coursework_assets.py`, you should have:

- `presentation/assets/slide1_hero_frame.png`
- `presentation/assets/slide2_concept_graphic.png`
- `presentation/assets/slide3_pipeline_graphic.png`
- `presentation/assets/slide4_training_progress.png`
- `presentation/assets/slide5_final_results.png`
- `presentation/assets/demo_frame_candidates.png`

These are intended to be inserted directly into PowerPoint.
