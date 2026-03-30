# Point Gather Environment Port

This folder contains a Safety-Gymnasium Point Gather port inspired by the Point-Gather task used in the CPO paper.

## Scope

This README describes only the environment port:

- task mechanics
- observations
- reward and cost behavior
- placement and termination
- similarities and differences from the original environment

**We were not able to closely replicate paper performance with this custom environment, this may be due to environment implementation so this should be sed with caution.**

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
- The current implementation therefore rescales the world with `spatial_scale = 0.05`.
- The current code uses `robot_object_spacing = 0.1`.
- The current code uses `catch_range = 0.15`.
- The current code uses `sensor_range = 0.6`.
- The current code uses `object_grid_scale = 0.1`.

## Registered Environments

- `SafetyPointGather0-v0`
- `SafetyPointGather1-v0`
- `SafetyPointGather2-v0`

For the paper-style CPO experiment, `SafetyPointGather1-v0` is the intended primary environment.
