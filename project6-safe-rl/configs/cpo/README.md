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

`config_point_gather.yaml` keeps the main Point-Gather algorithm settings close to the supplement and original script:

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
- The original implementation did not use OmniSafe defaults such as observation normalization, critic norm regularization, or linear LR decay. These were disabled where possible to stay closer to the original experiment.
- `vector_env_nums = 4` and `torch_threads = 1` are runtime choices benchmarked on my (chris') machine, change as needed.
- The Point Gather config runs on an adapted custom environment whose spatial scale was changed to fit Safety-Gymnasium's `Point` dynamics. Although the agent learns in this environement, we are not able to replicate the results on our custom gather environment. This should therefor be used with caution.
- The Point Circle config targets Safety-Gymnasium's native `SafetyPointCircle1-v0`, which is not the exact same environment implementation as the original paper's Circle task. However, the docs for Safety-Gymnasium cite the orignal CPO paper as inspiration for this environment so we believe this is a close adaptation.
- For comparison with other algorithms we report plots based on environement steps instead of "iters", but we trained with the equalivalent steps.

## Files

- `config_point_gather.yaml`: main CPO config for the paper-style Point Gather run
- `config_point_circle.yaml`: main CPO config for a native Safety-Gymnasium Point Circle run
x