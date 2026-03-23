# CPO Configuration Notes

This folder documents how the OmniSafe CPO configs are related to the original CPO paper experiments.

## Scope

This README describes only the algorithm/config side:

- CPO hyperparameters
- network architecture
- optimizer-related mappings
- similarities and differences between the original implementation and the OmniSafe version

## Reference

These configs were checked against:

- Achiam et al., "Constrained Policy Optimization" (ICML 2017)
- the experimental parameters in the paper supplement
- the original `jachiam/cpo` experiment scripts

Primary sources:

- Paper: https://proceedings.mlr.press/v70/achiam17a.html
- Supplement: https://proceedings.mlr.press/v70/achiam17a/achiam17a-supp.pdf
- Original repository: https://github.com/jachiam/cpo

## Point Gather Config: What Still Matches the Paper

`config_pointgather.yaml` keeps the main Point-Gather algorithm settings close to the supplement and original script:

- `seed = 1`
- `steps_per_epoch = 50000`
- total training length matched to `100` epochs
- effective parallel rollout count matched to `4`
- actor hidden sizes `(64, 32)`
- critic hidden sizes `(64, 32)`
- `tanh` activations
- `gamma = 0.995`
- `cost_gamma = 1.0`
- `lam = 0.95`
- `lam_c = 1.0`
- `target_kl = 0.01`
- `cost_limit = 0.1`
- `cg_iters = 10`
- `cg_damping = 1e-5`
- full-batch update behavior approximated with `update_iters = 1` and `batch_size = 50000`
- Fisher-vector-product subsampling approximated with `fvp_sample_freq = 5`, corresponding to roughly 20% of samples

## Point Circle Config: What Still Matches the Paper

`config_point_circle.yaml` keeps the main Point-Circle algorithm settings close to the supplement:

- `seed = 1`
- `steps_per_epoch = 50000`
- total training length matched to `100` epochs
- effective parallel rollout count matched to `4`
- actor hidden sizes `(64, 32)`
- critic hidden sizes `(64, 32)`
- `tanh` activations
- `gamma = 0.995`
- `cost_gamma = 1.0`
- `lam = 0.95`
- `lam_c = 1.0`
- `target_kl = 0.01`
- `cost_limit = 5.0`
- `cg_iters = 10`
- `cg_damping = 1e-5`
- full-batch update behavior approximated with `update_iters = 1` and `batch_size = 50000`

This config targets Safety-Gymnasium's native `SafetyPointCircle1-v0`, whose registered episode limit is `500` steps in the current install.

## Shared Differences From the Original Implementation

- The original code was built on rllab. This setup uses OmniSafe.
- The original code used `GaussianMLPPolicy` and separate `GaussianMLPBaseline` objects. OmniSafe uses its own actor-critic implementation.
- The original reward baseline and safety baseline used a conjugate-gradient regressor optimizer. OmniSafe critics are trained through its own critic update path instead of reproducing those exact baseline internals.
- The original optimizer code used `reg_coeff=1e-5` and `subsample_factor=0.2`. In OmniSafe, the closest mapping is `cg_damping=1e-5` and `fvp_sample_freq=5`.
- The original implementation did not use OmniSafe defaults such as observation normalization, critic norm regularization, or linear LR decay. These were disabled where possible to stay closer to the original experiment.
- Weight initialization is only approximated. OmniSafe does not expose the original rllab initializer stack directly, so the config uses the closest available built-in option.
- Advantage handling is matched as closely as possible through OmniSafe’s available settings, but the internal update pipeline is still OmniSafe’s implementation rather than the original one.
- `vector_env_nums = 4` and `torch_threads = 1` are runtime choices benchmarked on this machine. They are not paper-specified hyperparameters.
- The Point Gather config runs on an adapted custom environment whose spatial scale was changed to fit Safety-Gymnasium's `Point` dynamics.
- The Point Circle config targets Safety-Gymnasium's native `SafetyPointCircle1-v0`, which is not the exact same environment implementation as the original paper's Circle task.
- Safety-Gymnasium's native Point Circle environment uses its own episode-length and wrapper behavior, so the config is closer to the paper at the optimizer/workload level than at the environment level.
- The current Safety-Gymnasium install does not register `SafetyHumanoidCircle*`, so this folder does not currently provide a Humanoid-Circle CPO config.

## What This Means for Reproduction

These configs should be treated as:

- close to the original CPO experiments at the hyperparameter level
- not an exact port of the original algorithm implementation
- dependent on modern OmniSafe and Safety-Gymnasium environment behavior
- useful for best-effort reproduction attempts, but not enough on their own to claim paper-faithful results

For Point Gather specifically, the current config is paper-matched at the algorithm level, but the environment is now an adapted, scaled port rather than a same-scale copy of the original benchmark.

## Files

- `config_pointgather.yaml`: main CPO config for the paper-style Point Gather run
- `config_point_circle.yaml`: main CPO config for a native Safety-Gymnasium Point Circle run
