# CPO Configuration Notes

This folder documents how the OmniSafe CPO setup is matched to the original Point Gather experiment from the CPO paper.

## Scope

This README describes only the algorithm/config side:

- CPO hyperparameters
- network architecture
- optimizer-related mappings
- similarities and differences between the original implementation and the OmniSafe version

## Reference

This configuration was matched against:

- Achiam et al., "Constrained Policy Optimization" (ICML 2017)
- `jachiam/cpo/experiments/CPO_point_gather.py`
- supporting optimizer and safety-constraint code from the original repository

## Similarities to the Original CPO Point Gather Experiment

- `seed = 1`
- paper Point Gather environment target: `SafetyPointGather1-v0`
- `steps_per_epoch = 50000`
- total training length matched to `100` epochs
- effective parallel rollout count matched to `4`
- actor hidden sizes matched to `(64, 32)`
- baseline / critic hidden sizes matched to `(64, 32)`
- actor nonlinearity matched as closely as possible with `tanh`
- critic nonlinearity matched as closely as possible with `tanh`
- `gamma = 0.995`
- cost discount matched with `cost_gamma = 1.0`
- `lam = 0.95`
- `lam_c = 1.0`
- KL step target matched with `target_kl = 0.01`
- safety threshold matched with `cost_limit = 0.1`
- conjugate-gradient iterations matched with `cg_iters = 10`
- Fisher-vector-product subsampling approximated with `fvp_sample_freq = 5`, corresponding to roughly 20% of samples
- full-batch update behavior approximated with:
  `update_iters = 1` and `batch_size = 50000`

## Differences From the Original Implementation

- The original code was built on rllab. This setup uses OmniSafe.
- The original code used `GaussianMLPPolicy` and separate `GaussianMLPBaseline` objects. OmniSafe uses its own actor-critic implementation.
- The original reward baseline and safety baseline used a conjugate-gradient regressor optimizer. OmniSafe critics are trained through its own critic update path instead of reproducing those exact baseline internals.
- The original optimizer code used `reg_coeff=1e-5` and `subsample_factor=0.2`. In OmniSafe, the closest mapping is `cg_damping=1e-5` and `fvp_sample_freq=5`.
- The original implementation did not use OmniSafe defaults such as observation normalization, critic norm regularization, or linear LR decay. These were disabled where possible to stay closer to the original experiment.
- Weight initialization is only approximated. OmniSafe does not expose the original rllab initializer stack directly, so the config uses the closest available built-in option.
- Advantage handling is matched as closely as possible through OmniSafe’s available settings, but the internal update pipeline is still OmniSafe’s implementation rather than the original one.

## What This Means for Reproduction

This config should be treated as:

- close to the original Point Gather CPO experiment at the hyperparameter level
- not an exact port of the original algorithm implementation
- a best-effort OmniSafe translation designed for paper-result reproduction rather than codebase reproduction

## Files

- `config_pointgather.yaml`: main CPO config for the paper-style Point Gather run
