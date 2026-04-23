"""
run_ppopidlag.py — Grid runner for OmniSafe CPPOPID (PPO + PID-Lagrangian).

Mirrors the pattern used by run_ppopidshield_grid.py and run_dmbppo_grid.py:
calls omnisafe.Agent directly for each (env_id, seed) pair so that any YAML
config value — including nested dicts and lists such as hidden_sizes — is
forwarded correctly.  (ExperimentGrid.add() requires hashable values and
cannot accept lists or nested dicts.)

OmniSafe writes its own output directory:
    runs/CPPOPID-{env_id}/seed-{seed:03d}-{timestamp}/progress.csv

Usage examples
--------------
# Single env, single seed (quick test):
    python scripts/run_ppopidlag.py \\
        --envs SafetyPointGoal2-v0 \\
        --seeds 0 \\
        --config configs/cppopid/config.yaml

# Two environments × three seeds, sequential:
    python scripts/run_ppopidlag.py \\
        --envs SafetyPointGoal1-v0 SafetyPointGoal2-v0 \\
        --seeds 0 1 2 \\
        --config configs/cppopid/config.yaml

# Parallel across 4 worker processes:
    python scripts/run_ppopidlag.py \\
        --envs SafetyPointGoal2-v0 \\
        --seeds 0 1 2 \\
        --num_pool 4 \\
        --config configs/cppopid/config.yaml
"""

import argparse
import multiprocessing as mp
import time
import warnings
from pathlib import Path

import yaml


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker(job: dict) -> dict:
    """Run a single (env_id, seed) CPPOPID experiment and return a summary."""
    import sys
    project_root = job['project_root']
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import omnisafe

    env_id = job['env_id']
    seed   = job['seed']
    cfg    = dict(job['cfg'])   # shallow copy; OmniSafe mutates it
    cfg['seed'] = seed

    print(f"\n[CPPOPID grid] env={env_id}  seed={seed}", flush=True)
    t0 = time.time()

    agent = omnisafe.Agent('CPPOPID', env_id, custom_cfgs=cfg)
    reward, cost, _ = agent.learn()

    elapsed = time.time() - t0
    summary = dict(
        env_id=env_id, seed=seed,
        elapsed_sec=round(elapsed, 1),
        final_ret=reward,
        final_cost=cost,
    )
    print(
        f"[CPPOPID grid] DONE  env={env_id}  seed={seed}  "
        f"({elapsed/60:.1f} min)  ret={reward:.2f}  cost={cost:.2f}",
        flush=True,
    )
    return summary


# ── Grid runner ───────────────────────────────────────────────────────────────

def run_grid(
    envs:     list[str],
    seeds:    list[int],
    cfg:      dict,
    num_pool: int = 1,
) -> list[dict]:
    jobs = [
        dict(
            env_id=env_id,
            seed=seed,
            cfg=cfg,
            project_root=str(Path(__file__).parent.parent.resolve()),
        )
        for env_id in envs
        for seed in seeds
    ]

    n = len(jobs)
    print(f"\n{'='*60}")
    print(f"  CPPOPID (PPO + OmniSafe PID-Lagrangian) ExperimentGrid")
    print(f"  Environments : {envs}")
    print(f"  Seeds        : {seeds}")
    print(f"  Total jobs   : {n}   |   workers: {num_pool}")
    print(f"{'='*60}\n")

    if num_pool > 1:
        n_cpu = mp.cpu_count()
        if num_pool > n_cpu:
            warnings.warn(
                f"num_pool={num_pool} exceeds CPU count ({n_cpu}). "
                "Consider lowering to avoid memory contention.",
                stacklevel=2,
            )
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=min(num_pool, n)) as pool:
            summaries = pool.map(_worker, jobs)
    else:
        summaries = [_worker(j) for j in jobs]

    print(f"\n{'='*60}")
    print(f"  Grid complete — {n} runs finished")
    print(f"{'='*60}")
    print(f"  {'env_id':<30}  {'seed':>4}  {'ret':>8}  {'cost':>8}  {'min':>6}")
    for s in summaries:
        print(
            f"  {s['env_id']:<30}  {s['seed']:>4}  "
            f"{s['final_ret']:>8.2f}  {s['final_cost']:>8.2f}  "
            f"{s['elapsed_sec']/60:>6.1f}"
        )

    return summaries


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run CPPOPID over an experiment grid of environments and seeds.'
    )
    parser.add_argument(
        '--envs', nargs='+',
        default=['SafetyPointGoal2-v0'],
        help='Space-separated list of Safety-Gymnasium env IDs',
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int,
        default=[0],
        help='Space-separated list of random seeds',
    )
    parser.add_argument(
        '--config',
        default='configs/cppopid/config.yaml',
        help='Path to CPPOPID YAML config',
    )
    parser.add_argument(
        '--num_pool', type=int, default=1,
        help='Number of parallel worker processes (default: 1 = sequential)',
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Strip top-level keys that are per-job, not shared config
    cfg.pop('algo',   None)
    cfg.pop('env_id', None)
    cfg.pop('seed',   None)

    run_grid(
        envs=args.envs,
        seeds=args.seeds,
        cfg=cfg,
        num_pool=args.num_pool,
    )


# Examples:
#   python scripts/run_ppopidlag.py --envs SafetyPointGoal2-v0 --seeds 0
#   python scripts/run_ppopidlag.py --envs SafetyPointGoal2-v0 --seeds 0 1 2 --num_pool 3

if __name__ == '__main__':
    mp.freeze_support()
    main()
