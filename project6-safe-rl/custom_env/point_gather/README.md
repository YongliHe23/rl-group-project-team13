# Point Gather Environment Port

This folder contains a Safety-Gymnasium Point Gather port inspired by the Point-Gather task used in the CPO paper.

## Scope

This README describes only the environment port:

- task mechanics
- observations
- reward and cost behavior
- placement and termination
- similarities and differences from the original environment

## Environment Files

- `point_gather_task.py`: main Point Gather task implementation
- `register_env.py`: registers `SafetyPointGather0-v0`, `SafetyPointGather1-v0`, and `SafetyPointGather2-v0`

## Intended Reference

This implementation was checked against:

- Achiam et al., "Constrained Policy Optimization" (ICML 2017)
- the experimental parameters in the paper supplement
- the original `jachiam/cpo` Point-Gather experiment and environment code

Primary sources:

- Paper: https://proceedings.mlr.press/v70/achiam17a.html
- Supplement: https://proceedings.mlr.press/v70/achiam17a/achiam17a-supp.pdf
- Original repository: https://github.com/jachiam/cpo

The goal of this port is to preserve the benchmark's reward-cost structure and observation style while remaining trainable on top of Safety-Gymnasium's built-in `Point` robot.

## Paper-Matched Task Semantics

- The main paper setting uses `2` apples and `8` bombs for Point-Gather.
- Apple reward is `+10`.
- Bomb events contribute a reward penalty of `-1` and a safety cost of `1`.
- Episode horizon is `15` steps.
- Sensor layout uses `10` bins over a semicircle with `sensor_span = pi`.
- Observations are the robot sensor state followed by apple sensor bins and bomb sensor bins.
- Sensor intensities use the same distance form `1 - dist / sensor_range`.
- Objects are removed once collected.
- Episodes terminate when all remaining objects are gone or the `15`-step limit is reached.
- Object placement still follows the original grid-sampling pattern, but on a smaller spatial scale.

## Important Adaptations Relative to the Original Benchmark

- The original environment was built on an older rllab / MuJoCo stack. This port is implemented as a Safety-Gymnasium `BaseTask`.
- Safety-Gymnasium's built-in `Point` robot uses forward-and-turn control, not the exact original point-agent dynamics.
- Because of that dynamics mismatch, the original Point-Gather spatial constants were not reachable in `15` steps on the Safety-Gymnasium robot.
- The current implementation therefore rescales the world with `spatial_scale = 0.05`.
- The current code uses `robot_object_spacing = 0.1`.
- The current code uses `catch_range = 0.15`.
- The current code uses `sensor_range = 0.6`.
- The current code uses `object_grid_scale = 0.1`.
- The agent is explicitly reset at the origin with heading `0.0` so the gather layout is centered and consistent across episodes.
- The implementation keeps gather objects in task state rather than recreating apples and bombs as visible MuJoCo world geoms.
- The environment uses a fast reset path after the first world build, resetting MuJoCo state in place instead of rebuilding the entire world every episode.
- The safety signal is exposed in Safety-Gymnasium / OmniSafe style through `cost_bombs` and `cost_sum` instead of the original `env_infos['bombs']` interface.

## What This Means for Reproduction

This environment should be treated as:

- faithful to the Point-Gather reward, cost, horizon, and observation structure used in the paper
- faithful to the paper's main object counts for the Point-Gather setting
- adapted in physical scale so that the task is reachable under Safety-Gymnasium's current `Point` dynamics
- not a same-scale reproduction of the original `mujoco_safe` Point-Gather world
- not an exact quantitative reproduction of the original paper environment

In practice, this means it is a good adapted benchmark for comparing algorithms inside the current Safety-Gymnasium / OmniSafe stack, but claims of reproducing the original paper's Point-Gather results should be made carefully.

## Registered Environments

- `SafetyPointGather0-v0`
- `SafetyPointGather1-v0`
- `SafetyPointGather2-v0`

For the paper-style CPO experiment, `SafetyPointGather1-v0` is the intended primary environment.
