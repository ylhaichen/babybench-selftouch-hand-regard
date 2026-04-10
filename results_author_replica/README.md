# Author-Style Self-Touch Results

This directory contains the lightweight artifacts for the `self-touch` coursework run in `BabyBench`.

The training pipeline used three stages:

- `base`: `5.0M` timesteps
- `difficult`: `2.0M` timesteps
- `after`: `2.0M` timesteps

Large files such as `weights`, `checkpoints`, `vecnormalize.pkl`, `videos`, and zip archives are intentionally excluded from version control.

## Final Model Choice

The final reported model is the `after-stage` checkpoint.

Reason:

- it performed better than the `difficult-stage` checkpoint under standard deterministic evaluation
- it is the checkpoint used in the presentation material

## Key Results

### Training summary

| Stage | Timesteps | Episode return | Left / Right geoms |
|---|---:|---:|---:|
| Base | 5.0M | -13.13 | 10 / 11 |
| Difficult | 2.0M | 2.07 | 32 / 34 |
| After | 2.0M | 2.65 | 18 / 10 |

### Deterministic evaluation

`After-stage final model`

- `10` episodes
- `1000` steps per episode
- cumulative manual self-touch score: `0.2206`
- left touched geoms: `9`
- right touched geoms: `6`

`Difficult-stage final model`

- same evaluation protocol
- cumulative manual self-touch score: `0.0588`
- left touched geoms: `2`
- right touched geoms: `2`

## Tracked Result Files

The tracked files in this directory are intentionally lightweight:

- `live_metrics.csv`
- `live_metrics.png`
- `training.pkl`
- this summary README

These are enough to document the training dynamics and final selection without committing heavyweight artifacts.
