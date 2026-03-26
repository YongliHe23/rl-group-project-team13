"""
GCIQL (Goal-Conditioned Implicit Q-Learning)

Environment  : PointMaze – navigate variant (default; all OGBench state-based envs supported)
Dataset      : pointmaze-medium-navigate-v0 (default)

Training protocol (Table 2 reproduction):
  - 4 seeds (seeds 0-3)
  - 500 000 gradient steps per seed
  - Evaluate on all 5 test-time goals every 100 000 steps (at 100k, 200k, 300k, 400k, 500k)
  - Each evaluation: 50 rollout episodes per goal
  - Final reported success rate: average of the last 3 evaluations (300k, 400k, 500k)
  - Report mean ± std across seeds

References:
  - GCIQL paper : https://arxiv.org/abs/2110.06169  (IQL, Kostrikov et al. 2021)
  - OGBench     : https://arxiv.org/abs/2410.20092
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np

# ── OGBench impls path ────────────────────────────────────────────────────────
# Defaults to ../../../../ogbench/impls relative to this file.
# Override by setting the OGBENCH_IMPLS environment variable.
OGBENCH_IMPLS = os.environ.get(
    'OGBENCH_IMPLS',
    os.path.join(os.path.dirname(__file__), '../../../../ogbench/impls'),
)
sys.path.insert(0, os.path.abspath(OGBENCH_IMPLS))

import ml_collections
import ogbench
from tqdm import tqdm

from agents.gciql import GCIQLAgent, get_config as get_agent_config
from utils.datasets import Dataset, GCDataset
from utils.evaluation import evaluate

from config_gciql import get_config

# ─────────────────────────────── fixed constants ──────────────────────────────
DATASET_DIR   = '~/.ogbench/data'
EVAL_INTERVAL = 100_000            # evaluate every this many steps
FINAL_WINDOW  = 3                  # average last N evals for the final score
TASK_IDS      = list(range(1, 6))  # 5 test-time goals


# ──────────────────────────────── helpers ─────────────────────────────────────

def build_agent_config(cfg) -> ml_collections.ConfigDict:
    """Convert EnvConfig -> ml_collections.ConfigDict expected by GCIQLAgent."""
    agent_cfg = get_agent_config()
    agent_cfg.unlock()
    agent_cfg.actor_loss         = cfg.actor_loss
    agent_cfg.alpha              = cfg.alpha
    agent_cfg.discount           = cfg.discount
    agent_cfg.tau                = cfg.tau
    agent_cfg.expectile          = cfg.expectile
    agent_cfg.gc_negative        = cfg.gc_negative
    agent_cfg.actor_hidden_dims  = cfg.actor_hidden_dims
    agent_cfg.value_hidden_dims  = cfg.value_hidden_dims
    agent_cfg.layer_norm         = cfg.layer_norm
    agent_cfg.lr                 = cfg.lr
    agent_cfg.batch_size         = cfg.batch_size
    agent_cfg.value_p_curgoal    = cfg.value_p_curgoal
    agent_cfg.value_p_trajgoal   = cfg.value_p_trajgoal
    agent_cfg.value_p_randomgoal = cfg.value_p_randomgoal
    agent_cfg.value_geom_sample  = cfg.value_geom_sample
    agent_cfg.actor_p_curgoal    = cfg.actor_p_curgoal
    agent_cfg.actor_p_trajgoal   = cfg.actor_p_trajgoal
    agent_cfg.actor_p_randomgoal = cfg.actor_p_randomgoal
    agent_cfg.actor_geom_sample  = cfg.actor_geom_sample
    agent_cfg.const_std          = cfg.const_std
    agent_cfg.discrete           = cfg.discrete
    agent_cfg.encoder            = cfg.encoder
    agent_cfg.p_aug              = cfg.p_aug
    agent_cfg.frame_stack        = cfg.frame_stack
    agent_cfg.lock()
    return agent_cfg


def eval_all_tasks(agent, env, agent_cfg, num_episodes: int) -> tuple[float, list]:
    """Evaluate on all 5 test-time goals.

    Returns:
        avg_success : float, average success rate across all 5 goals
        per_task    : list of 5 floats, success rate per task
    """
    per_task = []
    for task_id in TASK_IDS:
        stats, _, _ = evaluate(
            agent=agent,
            env=env,
            task_id=task_id,
            config=agent_cfg,
            num_eval_episodes=num_episodes,
            num_video_episodes=0,
            eval_temperature=0,
            eval_gaussian=None,
        )
        per_task.append(float(stats['success']))
    avg_success = float(np.mean(per_task))
    return avg_success, per_task


# ─────────────────────────────────── main ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',         default='pointmaze', help='Environment base name')
    parser.add_argument('--task',        default='navigate',  help='Task type')
    parser.add_argument('--dsize',       default='medium',    help='Dataset size/difficulty')
    parser.add_argument('--train-steps', default=None, type=int,
                        help='Override number of gradient steps (default: 500 000)')
    parser.add_argument('--seeds',       default=4, type=int,
                        help='Number of seeds to run (default: 4)')
    parser.add_argument('--slurm-tqdm',  default=True,
                        action=argparse.BooleanOptionalAction,
                        help='Print every 10 000 steps instead of tqdm bar (use on HPC/Slurm)')
    args = parser.parse_args()

    env_name    = f'{args.env}-{args.dsize}-{args.task}-v0'
    cfg         = get_config(env_name)
    train_steps = args.train_steps if args.train_steps is not None else 500_000
    seeds       = list(range(args.seeds))
    n_evals     = train_steps // EVAL_INTERVAL

    print(f"\n{'='*60}")
    print(f'GCIQL on OGBench: {env_name}')
    print(f'  Train steps   : {train_steps:,}  |  Batch size : {cfg.batch_size}')
    print(f'  LR={cfg.lr}  γ={cfg.discount}  τ={cfg.tau}  κ={cfg.expectile}')
    print(f'  actor_loss={cfg.actor_loss}  α={cfg.alpha}')
    print(f'  Hidden dims   : actor={cfg.actor_hidden_dims}  value={cfg.value_hidden_dims}')
    print(f'  Value goals   : p_cur={cfg.value_p_curgoal}  '
          f'p_traj={cfg.value_p_trajgoal}  p_rand={cfg.value_p_randomgoal}')
    print(f'  Actor goals   : p_traj={cfg.actor_p_trajgoal}  '
          f'p_rand={cfg.actor_p_randomgoal}')
    print(f'  Seeds         : {seeds}')
    print(f'  Evals         : every {EVAL_INTERVAL:,} steps  '
          f'({n_evals} total)  |  final window: last {FINAL_WINDOW}')
    print(f"{'='*60}\n")

    # ── Load environment + dataset (shared across seeds) ──────────────────────
    print('Loading environment and dataset ...')
    env, train_raw, val_raw = ogbench.make_env_and_datasets(
        env_name,
        dataset_dir=DATASET_DIR,
        compact_dataset=False,
    )
    print(f'  Train size : {len(train_raw["observations"]):,}')
    print(f'  Obs dim    : {train_raw["observations"].shape[1]}')
    print(f'  Action dim : {train_raw["actions"].shape[1]}\n')

    # ml_collections config (same for all seeds; only JAX PRNGKey differs)
    agent_cfg = build_agent_config(cfg)

    # ── Multi-seed training loop ───────────────────────────────────────────────
    seed_final_scores = []   # one scalar per seed

    for seed in seeds:
        print(f'\n{"─"*50}')
        print(f'Seed {seed}/{seeds[-1]}')
        print(f'{"─"*50}')

        # Fresh agent and dataset wrapper for this seed
        agent = GCIQLAgent.create(
            seed=seed,
            ex_observations=train_raw['observations'][:1],
            ex_actions=train_raw['actions'][:1],
            config=agent_cfg,
        )
        train_dataset = GCDataset(Dataset(train_raw), agent_cfg)

        checkpoint_scores = []   # avg success at each eval checkpoint

        if args.slurm_tqdm:
            # ── HPC mode: plain print every 10k steps ─────────────────────────
            for step in range(1, train_steps + 1):
                batch    = train_dataset.sample(cfg.batch_size)
                agent, _ = agent.update(batch)

                if step % 10_000 == 0:
                    print(f'  step {step:>8,}/{train_steps:,}', flush=True)

                if step % EVAL_INTERVAL == 0:
                    print(f'  [eval @ step {step:,}]', flush=True)
                    avg, per_task = eval_all_tasks(agent, env, agent_cfg,
                                                   cfg.eval_episodes)
                    per_task_str  = '  '.join(
                        f't{i+1}={s:.2f}' for i, s in enumerate(per_task)
                    )
                    print(f'    [{per_task_str}]  avg={avg:.4f}', flush=True)
                    checkpoint_scores.append(avg)
        else:
            # ── Interactive mode: tqdm bar ─────────────────────────────────────
            pbar = tqdm(range(1, train_steps + 1),
                        desc=f'Seed {seed}', unit='step')
            for step in pbar:
                batch        = train_dataset.sample(cfg.batch_size)
                agent, _     = agent.update(batch)

                if step % EVAL_INTERVAL == 0:
                    avg, per_task = eval_all_tasks(agent, env, agent_cfg,
                                                   cfg.eval_episodes)
                    checkpoint_scores.append(avg)
                    pbar.set_postfix(
                        eval=f'{avg:.3f}',
                        n=len(checkpoint_scores),
                    )

        # Final score = average of last FINAL_WINDOW checkpoints
        final_score = float(np.mean(checkpoint_scores[-FINAL_WINDOW:]))
        seed_final_scores.append(final_score)

        ckpt_str = '  '.join(
            f'{(i+1)*EVAL_INTERVAL//1000}k={s:.3f}'
            for i, s in enumerate(checkpoint_scores)
        )
        print(f'  Checkpoints : {ckpt_str}')
        print(f'  Final score (avg of last {FINAL_WINDOW}): {final_score:.4f}')

    # ── Final report across seeds ──────────────────────────────────────────────
    mean_pct = float(np.mean(seed_final_scores)) * 100
    std_pct  = float(np.std(seed_final_scores))  * 100

    print(f'\n{"="*60}')
    print(f'Results on {env_name}')
    print(f'  Per-seed scores: '
          + '  '.join(f's{s}={v*100:.1f}%' for s, v in zip(seeds, seed_final_scores)))
    print(f'  Success rate: {mean_pct:.1f} ± {std_pct:.1f}%  '
          f'(last {FINAL_WINDOW} evals avg, {args.seeds} seeds)')
    print(f'{"="*60}')

    # ── Save results ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    out_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f'results_{env_name}_{timestamp}.txt')
    with open(out_path, 'w') as f:
        f.write(f'env={env_name}  train_steps={train_steps}  seeds={seeds}\n')
        f.write(f'actor_loss={cfg.actor_loss}  alpha={cfg.alpha}  '
                f'discount={cfg.discount}  expectile={cfg.expectile}\n')
        f.write(f'actor_p_traj={cfg.actor_p_trajgoal}  '
                f'actor_p_rand={cfg.actor_p_randomgoal}\n')
        f.write(f'eval_interval={EVAL_INTERVAL}  final_window={FINAL_WINDOW}\n')
        f.write('\nper_seed_final_scores:\n')
        for s, v in zip(seeds, seed_final_scores):
            f.write(f'  seed{s}: {v * 100:.2f}%\n')
        f.write(f'\nsuccess_rate: {mean_pct:.2f} ± {std_pct:.2f}%\n')
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
