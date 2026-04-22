"""
run_ppopidshield_grid.py — Experiment grid runner for PPO-PIDShield.

Mirrors run_experiment_grid.py (OmniSafe's ExperimentGrid pattern) for the
custom PPO-PIDShield algorithm.  Runs the algorithm across a configurable list
of environments and seeds, saving progress CSVs in the same structure used by
OmniSafe and the other custom algorithms:

    runs/PPOPIDShield-{env_id}/seed-{seed:03d}-{timestamp}/progress.csv

Each progress.csv uses OmniSafe-compatible column names so that
plot_ppopidshield.py and plot_comparison.py can consume it without conversion.

Usage examples
--------------
# Single env, single seed (quick test):
    python scripts/run_ppopidshield_grid.py \\
        --envs SafetyPointGoal1-v0 \\
        --seeds 0 \\
        --config configs/ppopidshield/config.yaml

# Two environments × three seeds, run sequentially:
    python scripts/run_ppopidshield_grid.py \\
        --envs SafetyPointGoal1-v0 SafetyCarGoal1-v0 \\
        --seeds 0 1 2 \\
        --config configs/ppopidshield/config.yaml

# Parallel across 4 worker processes (mirrors ExperimentGrid num_pool):
    python scripts/run_ppopidshield_grid.py \\
        --envs SafetyPointGoal1-v0 SafetyCarGoal1-v0 \\
        --seeds 0 1 2 \\
        --num_pool 4 \\
        --config configs/ppopidshield/config.yaml
"""

import argparse
import csv
import multiprocessing as mp
import time
import warnings
from datetime import datetime
from pathlib import Path

import yaml


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker(job: dict) -> dict:
    """Run a single (env_id, seed) PPO-PIDShield experiment; return a summary."""
    import sys
    project_root = job['project_root']
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from scripts.train_ppopidshield import PPOPIDShieldTrainer

    env_id   = job['env_id']
    seed     = job['seed']
    cfg      = job['cfg']
    runs_dir = Path(job['runs_dir'])

    timestamp  = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_subdir = runs_dir / f"PPOPIDShield-{{{env_id}}}" / f"seed-{seed:03d}-{timestamp}"
    run_subdir.mkdir(parents=True, exist_ok=True)

    progress_csv = run_subdir / 'progress.csv'

    print(f"\n[PPO-PIDShield grid] env={env_id}  seed={seed}  → {run_subdir}",
          flush=True)
    t0 = time.time()

    trainer = PPOPIDShieldTrainer(env_id, cfg, seed)
    rows    = trainer.train()

    # Write OmniSafe-compatible progress.csv
    omnisafe_rows = []
    for r in rows:
        omnisafe_rows.append({
            'TotalEnvSteps':     r['TotalEnvSteps'],
            'Metrics/EpRet':     r['Metrics_EpRet'],
            'Metrics/EpCost':    r['Metrics_EpCost'],
            'Metrics/EpLen':     r['Metrics_EpLen'],
            'Train/Lambda':      r['Train_Lambda'],
            'Train/CurrLimit':   r['Train_CurrLimit'],
            'epoch':             r['epoch'],
            'kl':                r['kl'],
            'n_eps':             r['n_eps'],
        })

    with open(progress_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=omnisafe_rows[0].keys())
        writer.writeheader()
        writer.writerows(omnisafe_rows)

    elapsed = time.time() - t0
    summary = dict(
        env_id=env_id, seed=seed,
        run_dir=str(run_subdir), progress_csv=str(progress_csv),
        elapsed_sec=round(elapsed, 1),
        final_ret=rows[-1]['Metrics_EpRet'],
        final_cost=rows[-1]['Metrics_EpCost'],
    )
    print(f"[PPO-PIDShield grid] DONE  env={env_id}  seed={seed}  "
          f"({elapsed/60:.1f} min)  "
          f"ret={summary['final_ret']:.2f}  cost={summary['final_cost']:.2f}",
          flush=True)
    return summary


# ── Grid runner ───────────────────────────────────────────────────────────────

def run_grid(
    envs:     list[str],
    seeds:    list[int],
    cfg:      dict,
    runs_dir: str = 'runs',
    num_pool: int = 1,
) -> list[dict]:
    """
    Run PPO-PIDShield for every (env_id, seed) combination.
    Mirrors ExperimentGrid.run() interface for consistency.

    Parameters
    ----------
    envs:     list of Safety-Gymnasium environment IDs
    seeds:    list of random seeds
    cfg:      loaded YAML config dict (env_id / seed / algo keys stripped per job)
    runs_dir: root output directory — mirrors OmniSafe default
    num_pool: number of parallel worker processes (1 = sequential)
    """
    jobs = []
    for env_id in envs:
        for seed in seeds:
            job_cfg = {k: v for k, v in cfg.items()
                       if k not in ('env_id', 'seed', 'algo')}
            jobs.append(dict(
                env_id=env_id, seed=seed, cfg=job_cfg,
                runs_dir=runs_dir,
                project_root=str(Path(__file__).parent.parent.resolve()),
            ))

    n = len(jobs)
    print(f"\n{'='*60}")
    print(f"  PPO-PIDShield ExperimentGrid")
    print(f"  Environments : {envs}")
    print(f"  Seeds        : {seeds}")
    print(f"  Total jobs   : {n}   |   workers: {num_pool}")
    print(f"  Output root  : {runs_dir}")
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
        print(f"  {s['env_id']:<30}  {s['seed']:>4}  "
              f"{s['final_ret']:>8.2f}  {s['final_cost']:>8.2f}  "
              f"{s['elapsed_sec']/60:>6.1f}")

    return summaries


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run PPO-PIDShield over an experiment grid of environments and seeds.'
    )
    parser.add_argument(
        '--envs', nargs='+',
        default=['SafetyPointGoal1-v0'],
        help='Space-separated list of Safety-Gymnasium env IDs',
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int,
        default=[0],
        help='Space-separated list of random seeds',
    )
    parser.add_argument(
        '--config',
        default='configs/ppopidshield/config.yaml',
        help='Path to PPO-PIDShield YAML config',
    )
    parser.add_argument(
        '--runs_dir',
        default='runs',
        help='Root directory for output CSVs (default: runs/)',
    )
    parser.add_argument(
        '--num_pool', type=int, default=1,
        help='Number of parallel worker processes (default: 1 = sequential)',
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_grid(
        envs=args.envs,
        seeds=args.seeds,
        cfg=cfg,
        runs_dir=args.runs_dir,
        num_pool=args.num_pool,
    )


# Example:
#   python scripts/run_ppopidshield_grid.py --envs SafetyPointGoal1-v0 --seeds 0 1 2
#   python scripts/run_ppopidshield_grid.py --num_pool 4 --envs SafetyPointGoal1-v0 --seeds 0 1 2

if __name__ == '__main__':
    mp.freeze_support()
    main()
