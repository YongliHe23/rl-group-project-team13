"""
CRL (Contrastive RL) — OGBench canonical implementation

Uses the CRLAgent from the OGBench impls directory (JAX/Flax), following the
same invocation pattern as main_gcivl.py for GCIVLAgent.  All --flag
configurations and result-saving logic match main_crl.py exactly.

OGBENCH_IMPLS must point to the ogbench/impls directory that contains:
    agents/crl.py        – CRLAgent class
    utils/datasets.py    – Dataset, GCDataset
    utils/evaluation.py  – evaluate()

Set via environment variable or edit the fallback path below.

References:
  - CRL paper  : https://arxiv.org/abs/2206.07568
  - OGBench    : https://arxiv.org/abs/2410.20092
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime

import numpy as np
import ogbench
from tqdm import tqdm

# ── OGBench impls path ────────────────────────────────────────────────────────
OGBENCH_IMPLS = os.environ.get(
    'OGBENCH_IMPLS',
    '/home/yonglihe/ece567/project/ogbench/impls',
)
sys.path.insert(0, os.path.abspath(OGBENCH_IMPLS))

import ml_collections
from agents.crl import CRLAgent, get_config as get_agent_config
from utils.datasets import Dataset, GCDataset
from utils.evaluation import evaluate

from config_crl import get_config

# ─────────────────────────────── fixed constants ──────────────────────────────
DATASET_DIR         = '~/.ogbench/data'
EVAL_INTERVAL       = 100_000
DEFAULT_TRAIN_STEPS = 500_000
SEEDS               = [42, 0, 1, 2]
FINAL_WINDOW        = 3          # avg over last N eval checkpoints for final score
TASK_IDS            = list(range(1, 6))
EVAL_TEMPERATURE    = 0.3        # softmax temperature for powderworld (discrete)


# ──────────────────────────────── helpers ─────────────────────────────────────

def build_agent_config(
    cfg,
    is_discrete: bool = False,
    is_visual: bool = False,
) -> ml_collections.ConfigDict:
    """Map config_crl EnvConfig → ml_collections.ConfigDict for CRLAgent.

    Fields present in config_crl are written explicitly; all other fields
    retain their OGBench defaults from get_agent_config().
    """
    agent_cfg = get_agent_config()
    agent_cfg.unlock()

    # ── hyper-parameters from config_crl ──────────────────────────────────────
    agent_cfg.lr         = cfg.lr
    agent_cfg.batch_size = cfg.batch_size
    agent_cfg.discount   = cfg.discount
    agent_cfg.alpha      = cfg.alpha

    # ── goal-sampling probabilities ────────────────────────────────────────────
    agent_cfg.value_p_trajgoal    = cfg.value_p_trajgoal
    agent_cfg.value_p_randomgoal  = cfg.value_p_randomgoal
    agent_cfg.value_geom_sample   = cfg.value_geom_sample
    agent_cfg.actor_p_trajgoal    = cfg.actor_p_trajgoal
    agent_cfg.actor_p_randomgoal  = cfg.actor_p_randomgoal
    agent_cfg.actor_geom_sample   = cfg.actor_geom_sample

    # ── environment modality ───────────────────────────────────────────────────
    agent_cfg.discrete    = is_discrete
    agent_cfg.encoder     = 'impala_small' if is_visual else None
    agent_cfg.actor_loss  = cfg.actor_loss

    agent_cfg.lock()
    return agent_cfg


def eval_all_tasks(
    agent,
    env,
    agent_cfg: ml_collections.ConfigDict,
    num_episodes: int,
    eval_temperature: float = 0.0,
) -> tuple:
    """Evaluate over all 5 tasks; return (mean_success_rate, per_task_list)."""
    per_task = []
    for task_id in TASK_IDS:
        stats, _, _ = evaluate(
            agent=agent,
            env=env,
            task_id=task_id,
            config=agent_cfg,
            num_eval_episodes=num_episodes,
            num_video_episodes=0,
            eval_temperature=eval_temperature,
            eval_gaussian=None,
        )
        per_task.append(float(stats['success']))
    return float(np.mean(per_task)), per_task


# ──────────────────────────── parallel-seed worker ───────────────────────────

def _crl_c_seed_worker(kwargs: dict):
    """Run one seed of CRL (OGBench canonical) training in a subprocess.

    Device note: HPC V100s typically run in exclusive-process compute mode.
    To avoid cudaErrorDevicesUnavailable, parallel workers hide the GPU via
    CUDA_VISIBLE_DEVICES='' and force JAX to the CPU backend before the first
    XLA computation.  Evaluation (MuJoCo, CPU-bound) is the dominant cost
    (~90 % of wall time) and parallelises freely across CPU cores.
    """
    seed          = kwargs['seed']
    env_name      = kwargs['env_name']
    agent_cfg     = kwargs['agent_cfg']
    train_steps   = kwargs['train_steps']
    eval_intv     = kwargs['eval_intv']
    train_raw     = kwargs['train_dataset']
    eval_episodes = kwargs['eval_episodes']
    is_visual     = kwargs['is_visual']
    is_discrete   = kwargs['is_discrete']
    dataset_dir   = kwargs['dataset_dir']
    device        = kwargs.get('device', 'gpu')

    # Force JAX to CPU before any XLA operation (lazy backend init lets this work
    # even though JAX was already imported at module level).
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['JAX_PLATFORM_NAME']    = 'cpu'
    print(f"  [Seed {seed}] using device: {device}", flush=True)

    np.random.seed(seed)

    print(f"  [Seed {seed}] creating env (env_only, no dataset reload) ...", flush=True)
    env = ogbench.make_env_and_datasets(env_name, dataset_dir=dataset_dir, env_only=True)
    print(f"  [Seed {seed}] env ready, building agent ...", flush=True)

    ex_actions = train_raw['actions'][:1]
    if is_discrete:
        ex_actions = np.full_like(ex_actions, env.action_space.n - 1)
    agent = CRLAgent.create(
        seed=seed,
        ex_observations=train_raw['observations'][:1],
        ex_actions=ex_actions,
        config=agent_cfg,
    )
    train_dataset = GCDataset(Dataset(train_raw), agent_cfg)
    eval_temperature = EVAL_TEMPERATURE if is_visual else 0.0

    print(f"  [Seed {seed}] starting training loop (print every 10k steps) ...", flush=True)

    seed_evals = []
    _t_loop_start = time.time()
    for step in range(1, train_steps + 1):
        batch        = train_dataset.sample(agent_cfg.batch_size)
        agent, _info = agent.update(batch)

        if step == 100:
            elapsed = time.time() - _t_loop_start
            ms_per_step = elapsed / 100 * 1000
            eta_h = ms_per_step * train_steps / 1000 / 3600
            print(f"  [Seed {seed}] step 100 done | {ms_per_step:.1f} ms/step | "
                  f"ETA ~{eta_h:.1f} h for {train_steps:,} steps", flush=True)

        if step % 10_000 == 0:
            print(f"  [Seed {seed}] step {step:>8,}/{train_steps:,}", flush=True)

        if step % eval_intv == 0:
            avg, per_task = eval_all_tasks(agent, env, agent_cfg,
                                           eval_episodes, eval_temperature)
            seed_evals.append((step, per_task, avg))
            task_str = "  ".join(f"T{i+1}:{sr*100:.1f}%" for i, sr in enumerate(per_task))
            print(f"\n  [Seed {seed}] step {step:,} | "
                  f"{task_str} | mean: {avg*100:.2f}%\n", flush=True)

    env.close()
    return (seed, seed_evals)


# ────────────────────────────────────── main ──────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",           default="pointmaze", help="Environment base name")
    parser.add_argument("--task",          default="navigate",  help="Task name")
    parser.add_argument("--dsize",         default="medium",    help="Dataset size/difficulty")
    parser.add_argument("--train-step",    default=None, type=int,
                        help=f"Gradient steps per seed (default: {DEFAULT_TRAIN_STEPS:,})")
    parser.add_argument("--eval-interval", default=EVAL_INTERVAL, type=int,
                        help="Evaluate every N steps (default: %(default)s)")
    parser.add_argument("--slurm-tqdm",    default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Print every 10 k steps instead of tqdm bar (Slurm mode)")
    parser.add_argument("--single-seed",   type=int, nargs="?", const=42, default=None,
                        metavar="SEED",
                        help="Run with a single seed (default 42 if flag given without value)")
    parser.add_argument("--seeds",         type=int, nargs="+", default=None,
                        metavar="SEED",
                        help="Explicit list of seeds to run sequentially, e.g. --seeds 42 0. "
                             "Ignored if --single-seed is set. "
                             "Recommended: split the default 4 seeds across 2 GPU jobs "
                             "(--seeds 42 0  in job1, --seeds 1 2  in job2) to stay within "
                             "MaxJobs=2 / GrpTRES=gres/gpu=2 Slurm limits while keeping "
                             "each job under the 8-hour wall-time limit.")
    parser.add_argument("--visual-enabled", action="store_true",
                        help="Enable visual encoder + discrete actor (required for "
                             "powderworld-* environments)")
    parser.add_argument("--parallel-seeds", action="store_true",
                        help="Run all seeds concurrently in separate processes "
                             "(each reloads the dataset; ~N-seed wall-clock speedup). "
                             "Offline RL has no env interaction during training, so "
                             "seeds are fully independent and safe to parallelise.")
    parser.add_argument("--output-dir",    default=".",
                        help="Directory to write the results .txt file "
                             "(default: current working directory). Created if absent.")
    args = parser.parse_args()

    env_name    = f"{args.env}-{args.dsize}-{args.task}-v0"
    cfg         = get_config(env_name)
    train_steps = args.train_step if args.train_step is not None else DEFAULT_TRAIN_STEPS
    eval_intv   = args.eval_interval

    if args.single_seed is not None:
        seeds = [args.single_seed]
    elif args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = SEEDS

    is_visual   = args.visual_enabled
    is_discrete = args.visual_enabled   # powderworld is always both visual and discrete

    agent_cfg        = build_agent_config(cfg, is_discrete=is_discrete, is_visual=is_visual)
    eval_temperature = EVAL_TEMPERATURE if is_visual else 0.0

    print(f"\n{'='*60}")
    print(f"CRL (OGBench canonical) on OGBench: {env_name}")
    print(f"  Seeds          : {seeds}"
          f"{'  [parallel]' if args.parallel_seeds and len(seeds) > 1 else ''}")
    print(f"  Train steps    : {train_steps:,} per seed  |  Eval every: {eval_intv:,}")
    print(f"  Batch size     : {agent_cfg.batch_size}  |  LR={agent_cfg.lr}  "
          f"gamma={agent_cfg.discount}  alpha={agent_cfg.alpha}")
    print(f"  Value goals    : geom(1-gamma) traj  (fixed)")
    print(f"  Actor goals    : p_traj={agent_cfg.actor_p_trajgoal}  "
          f"p_rand={agent_cfg.actor_p_randomgoal}")
    if is_visual:
        print(f"  Visual encoder : impala_small  |  Discrete actor "
              f"(temp={eval_temperature})")
    print(f"{'='*60}\n")

    # ── 1. Load environment and dataset ───────────────────────────────────────
    print("Loading environment and dataset ...")
    env, train_raw, _ = ogbench.make_env_and_datasets(
        env_name, dataset_dir=DATASET_DIR, compact_dataset=False,
    )
    obs = train_raw["observations"]
    action_info = ('(discrete)' if is_discrete
                   else str(train_raw["actions"].shape[1:]))
    print(f"  Train size : {len(obs):,}  |  obs: {obs.shape[1:]}  "
          f"actions: {action_info}"
          f"{'  (visual+discrete)' if is_visual else ''}\n")

    # ── 2. Multi-seed training ─────────────────────────────────────────────────
    # all_results[seed] = list of (step, per_task_success_list, overall_rate)
    all_results: dict = {}

    # Shared worker kwargs (seed-specific 'seed' key is filled per worker)
    _worker_base = dict(
        env_name      = env_name,
        agent_cfg     = agent_cfg,
        train_steps   = train_steps,
        eval_intv     = eval_intv,
        is_visual     = is_visual,
        is_discrete   = is_discrete,
        dataset_dir   = DATASET_DIR,
        eval_episodes = cfg.eval_episodes,
    )

    if args.parallel_seeds and len(seeds) > 1:
        # ── Parallel path: one subprocess per seed ────────────────────────────
        # Workers receive the already-loaded train_raw dict (avoids re-reading
        # from the shared HPC filesystem, which causes I/O contention).
        # CUDA_VISIBLE_DEVICES='' and JAX_PLATFORM_NAME=cpu are set inside each
        # worker before any XLA computation to avoid exclusive-process GPU mode
        # errors on HPC V100 nodes.
        env.close()
        worker_args = [{**_worker_base, 'seed': s, 'device': 'cpu',
                        'train_dataset': train_raw} for s in seeds]
        ctx = mp.get_context('spawn')
        print(f"Launching {len(seeds)} parallel seed workers "
              f"(spawn context, train_raw passed in-memory, device=cpu) ...\n")
        with ctx.Pool(processes=len(seeds)) as pool:
            results = pool.map(_crl_c_seed_worker, worker_args)
        all_results = dict(results)

    else:
        # ── Sequential path ────────────────────────────────────────────────────
        ex_actions = train_raw['actions'][:1]
        if is_discrete:
            ex_actions = np.full_like(ex_actions, env.action_space.n - 1)

        for seed_idx, seed in enumerate(seeds):
            print(f"\n{'─'*60}")
            print(f"Seed {seed}  ({seed_idx + 1}/{len(seeds)})")
            print(f"{'─'*60}")

            np.random.seed(seed)

            agent = CRLAgent.create(
                seed=seed,
                ex_observations=train_raw['observations'][:1],
                ex_actions=ex_actions,
                config=agent_cfg,
            )
            train_dataset = GCDataset(Dataset(train_raw), agent_cfg)

            seed_evals = []   # list of (step, per_task_list, overall_rate)

            if args.slurm_tqdm:
                _t_loop_start = time.time()
                for step in range(1, train_steps + 1):
                    batch        = train_dataset.sample(agent_cfg.batch_size)
                    agent, _info = agent.update(batch)

                    if step == 100:
                        elapsed = time.time() - _t_loop_start
                        ms_per_step = elapsed / 100 * 1000
                        eta_h = ms_per_step * train_steps / 1000 / 3600
                        print(f"  [Seed {seed}] step 100 done | {ms_per_step:.1f} ms/step | "
                              f"ETA ~{eta_h:.1f} h for {train_steps:,} steps")

                    if step % 10_000 == 0:
                        print(f"  step {step:>8,}/{train_steps:,}", flush=True)

                    if step % eval_intv == 0:
                        print(f"  [eval @ step {step:,}]", flush=True)
                        avg, per_task = eval_all_tasks(agent, env, agent_cfg,
                                                       cfg.eval_episodes,
                                                       eval_temperature)
                        seed_evals.append((step, per_task, avg))
                        task_str = "  ".join(
                            f"T{i+1}:{sr*100:.1f}%"
                            for i, sr in enumerate(per_task)
                        )
                        print(f"\n  [Seed {seed}] step {step:,} | "
                              f"{task_str} | mean: {avg*100:.2f}%\n")
            else:
                pbar = tqdm(range(1, train_steps + 1),
                            desc=f"Seed {seed}", unit="step")
                for step in pbar:
                    batch        = train_dataset.sample(agent_cfg.batch_size)
                    agent, _info = agent.update(batch)

                    if step % eval_intv == 0:
                        avg, per_task = eval_all_tasks(agent, env, agent_cfg,
                                                       cfg.eval_episodes,
                                                       eval_temperature)
                        seed_evals.append((step, per_task, avg))
                        pbar.write(f"  [Seed {seed}] step {step:,} | "
                                   f"mean: {avg*100:.2f}%")
                        pbar.set_postfix(eval=f"{avg:.3f}",
                                         n=len(seed_evals))

            all_results[seed] = seed_evals

        env.close()

    # ── 3. Summary statistics ──────────────────────────────────────────────────
    # Average over the last FINAL_WINDOW eval checkpoints.
    seed_avg_rates:    list = []
    seed_avg_per_task: list = []

    for seed in seeds:
        evals  = all_results[seed]
        last_w = evals[-FINAL_WINDOW:]
        seed_avg_rates.append(float(np.mean([e[2] for e in last_w])))
        n_tasks = len(last_w[0][1])
        seed_avg_per_task.append(
            [float(np.mean([e[1][t] for e in last_w])) for t in range(n_tasks)]
        )

    final_mean = float(np.mean(seed_avg_rates))
    final_std  = float(np.std(seed_avg_rates))

    n_tasks       = len(seed_avg_per_task[0])
    per_task_mean = [
        float(np.mean([seed_avg_per_task[i][t] for i in range(len(seeds))]))
        for t in range(n_tasks)
    ]
    per_task_std  = [
        float(np.std( [seed_avg_per_task[i][t] for i in range(len(seeds))]))
        for t in range(n_tasks)
    ]
    last_ckpts = [e[0] for e in all_results[seeds[0]][-FINAL_WINDOW:]]

    # ── 4. Print report ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS -- CRL (OGBench) -- {env_name}")
    print(f"Avg over last {FINAL_WINDOW} checkpoints: {last_ckpts}")
    print(f"{'='*60}")
    for seed, rate in zip(seeds, seed_avg_rates):
        print(f"  Seed {seed:>3}: {rate * 100:.2f}%")
    print(f"  {'─'*34}")
    print(f"  Mean +/- Std : {final_mean*100:.2f}% +/- {final_std*100:.2f}%")
    print(f"\nPer-task breakdown (mean +/- std across {len(seeds)} seeds):")
    for t, (m, s) in enumerate(zip(per_task_mean, per_task_std), start=1):
        print(f"  Task {t}: {m*100:.2f}% +/- {s*100:.2f}%")

    # ── 5. Save results ───────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    out_path  = os.path.join(
        args.output_dir,
        f"results_crl_c_{args.env}_{args.dsize}_{args.task}_{timestamp}.txt",
    )
    with open(out_path, "w") as f:
        f.write(f"method: CRL (OGBench canonical)\n")
        f.write(f"env: {env_name}\n")
        f.write(f"seeds: {seeds}\n")
        f.write(f"train_steps_per_seed: {train_steps}\n")
        f.write(f"eval_interval: {eval_intv}\n")
        f.write(f"final_window: {FINAL_WINDOW}\n")
        f.write(f"last_{FINAL_WINDOW}_checkpoints: {last_ckpts}\n")
        f.write(f"alpha={agent_cfg.alpha}  discount={agent_cfg.discount}\n")
        f.write(f"actor_p_traj={agent_cfg.actor_p_trajgoal}  "
                f"actor_p_rand={agent_cfg.actor_p_randomgoal}\n\n")

        for seed in seeds:
            f.write(f"Seed {seed}:\n")
            for step, per_task, overall in all_results[seed]:
                task_str = "  ".join(
                    f"T{i+1}:{sr*100:.1f}%" for i, sr in enumerate(per_task)
                )
                f.write(f"  step {step:>8,}: {task_str} | mean: {overall*100:.2f}%\n")
            idx = seeds.index(seed)
            f.write(f"  --> avg (last {FINAL_WINDOW} ckpts): "
                    f"{seed_avg_rates[idx]*100:.2f}%\n\n")

        f.write(f"Summary (avg over last {FINAL_WINDOW} checkpoints, "
                f"mean +/- std across {len(seeds)} seeds):\n")
        f.write(f"  Mean: {final_mean*100:.2f}%\n")
        f.write(f"  Std : {final_std*100:.2f}%\n\n")
        f.write(f"Per-task breakdown (mean +/- std across seeds):\n")
        for t, (m, s) in enumerate(zip(per_task_mean, per_task_std), start=1):
            f.write(f"  Task {t}: {m*100:.2f}% +/- {s*100:.2f}%\n")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
