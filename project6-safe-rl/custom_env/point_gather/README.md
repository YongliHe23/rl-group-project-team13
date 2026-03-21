# Point Gather Environment Port

This folder contains a Safety-Gymnasium port of the Point Gather environment used in the CPO paper.

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

This port is based on the Point Gather environment used in:

- Achiam et al., "Constrained Policy Optimization" (ICML 2017)
- original environment implementation in `jachiam/cpo/envs/mujoco/gather/gather_env.py`

The goal is behavioral fidelity for experiments, not source-level fidelity to the original rllab code.

## Similarities to the Original Environment

- The main paper setting uses `2` apples and `8` bombs.
- Apple reward is `+10`.
- Bomb penalty magnitude is `1`.
- Episode horizon is `15` steps.
- `activity_range=6`
- `robot_object_spacing=2`
- `catch_range=1`
- `n_bins=10`
- `sensor_range=6`
- `sensor_span=pi`
- Object placement follows the original grid-like sampling pattern.
- Objects are removed once collected.
- The episode terminates when all remaining objects are gone.
- Observation structure is robot-state features followed by apple sensor bins and bomb sensor bins.
- Sensor intensities use the original distance form `1 - dist / sensor_range`.
- Sensor visibility uses the original semicircle field of view.
- Sensor overwrite behavior approximates the original near-object occlusion logic by processing farther objects first.
- Bomb collection affects both reward and safety:
  reward gets `-bomb_cost`, and safety cost is exposed through `cost_bombs` / `cost_sum`.

## Differences From the Original Environment

- The original code was implemented as an rllab `GatherEnv` wrapper around an older MuJoCo environment. This port is implemented as a Safety-Gymnasium `BaseTask`.
- The original environment rendered apples and bombs as explicit MuJoCo world objects. This port reproduces the benchmark behavior in task logic rather than reproducing the original rendering path.
- The underlying point robot comes from modern Safety-Gymnasium rather than the original rllab point environment class.
- The original environment exposed bomb events through `env_infos['bombs']`. This port exposes the safety signal in OmniSafe/Safety-Gymnasium style as `cost_bombs` and `cost_sum`.
- The original environment included some wrapper-specific behavior around inner-environment termination. This port follows Safety-Gymnasium’s normal environment lifecycle instead of copying that wrapper logic exactly.

## What This Means for Reproduction

This port is intended to preserve the environment properties most likely to affect learning results:

- same task layout scale
- same object counts for the paper setting
- same apple and bomb event semantics
- same episode horizon
- similar observation content
- similar placement distribution
- same collect-and-remove dynamics

It should be treated as a paper-faithful environment reproduction at the task-behavior level, but not as an exact codebase port of the original repository.

## Registered Environments

- `SafetyPointGather0-v0`
- `SafetyPointGather1-v0`
- `SafetyPointGather2-v0`

For the paper-style CPO experiment, `SafetyPointGather1-v0` is the intended primary environment.
