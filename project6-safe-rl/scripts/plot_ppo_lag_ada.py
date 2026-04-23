"""Plot PPOLagAdapt runs with extra diagnostics for adaptive lambda schedules.

Usage
-----
# Auto-discover the latest PPOLagAdapt run under runs/:
    python scripts/plot_ppo_lag_ada.py

# Auto-discover the latest run for one schedule:
    python scripts/plot_ppo_lag_ada.py --schedule late_soft_adaptive

# Plot a specific run:
    python scripts/plot_ppo_lag_ada.py \\
        --csv runs/ppo_lag_adapt_late_soft_adaptive/seed-000-.../progress.csv

# Save to a custom output path:
    python scripts/plot_ppo_lag_ada.py --out figs/ppolag_adapt.png

# This also saves a comparison-layout figure alongside the diagnostic one:
#   figs/ppolag_adapt_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_COST_LIMIT = 25.0
DEFAULT_EPISODE_LEN = 1000.0


def repo_root() -> Path:
    """Return the project root."""
    return Path(__file__).resolve().parents[1]


def find_latest_progress(schedule: str | None = None) -> Path:
    """Find the latest PPOLagAdapt progress.csv."""
    runs_dir = repo_root() / 'runs'
    pattern = (
        f'ppo_lag_adapt_{schedule}/*/progress.csv'
        if schedule
        else 'ppo_lag_adapt_*/*/progress.csv'
    )
    paths = sorted(runs_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f'No progress.csv found under {runs_dir / pattern}. '
            'Run train_ppo_lag_ada.py first, or pass --csv explicitly.',
        )
    return paths[-1]


def find_lagrange_column(df: pd.DataFrame) -> str | None:
    """Return the best available lambda column."""
    candidates = [
        'Metrics/LagrangeMultiplier',
        'Metrics/LagrangeMultiplier/Mean',
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_run_metadata(progress_csv: Path) -> dict[str, Any]:
    """Read run metadata from config.json when available."""
    run_dir = progress_csv.parent
    meta: dict[str, Any] = {
        'run_dir': run_dir,
        'seed_tag': run_dir.name,
        'exp_name': run_dir.parent.name,
        'env_id': '',
        'schedule_name': '',
        'schedule_cfgs': {},
        'cost_limit': DEFAULT_COST_LIMIT,
    }

    config_path = run_dir / 'config.json'
    if not config_path.exists():
        return meta

    with config_path.open('r', encoding='utf-8') as handle:
        cfg = json.load(handle)

    meta['exp_name'] = cfg.get('exp_name', meta['exp_name'])
    meta['env_id'] = cfg.get('env_id', '')
    meta['schedule_cfgs'] = cfg.get('lambda_schedule_cfgs', {})
    meta['schedule_name'] = meta['schedule_cfgs'].get('lambda_schedule', '')
    meta['cost_limit'] = float(cfg.get('lagrange_cfgs', {}).get('cost_limit', DEFAULT_COST_LIMIT))
    return meta


def load_and_enrich(csv_path: Path, cost_limit: float) -> pd.DataFrame:
    """Load a progress CSV and add derived columns used by the plots."""
    df = pd.read_csv(csv_path).copy()

    required_cols = ['TotalEnvSteps', 'Metrics/EpRet', 'Metrics/EpCost', 'Metrics/EpLen']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f'Missing required columns in {csv_path}: {missing}')

    df['DeltaSteps'] = df['TotalEnvSteps'].diff().fillna(df['TotalEnvSteps'])
    df['CostRate_epoch_est'] = df['Metrics/EpCost'] / df['Metrics/EpLen'].clip(lower=1.0)
    df['CumulativeCost_est'] = (df['DeltaSteps'] * df['CostRate_epoch_est']).cumsum()
    df['CostRate_est'] = df['CumulativeCost_est'] / df['TotalEnvSteps'].clip(lower=1.0)
    df['CostViolation'] = df['Metrics/EpCost'] - cost_limit
    return df


def smooth(series: pd.Series, window: int) -> pd.Series:
    """Simple rolling mean with a warm start."""
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def print_run_summary(df: pd.DataFrame, meta: dict[str, Any], lag_col: str | None) -> None:
    """Print a compact summary of the plotted run."""
    final_steps = float(df['TotalEnvSteps'].iloc[-1])
    print(f'Run directory: {meta["run_dir"]}')
    if meta['env_id']:
        print(f'Environment : {meta["env_id"]}')
    if meta['schedule_name']:
        print(f'Schedule    : {meta["schedule_name"]}')
    print(f'Final steps : {final_steps:.0f}')

    if 'Time/Total' in df.columns:
        total_sec = float(df['Time/Total'].iloc[-1])
        hours = int(total_sec // 3600)
        minutes = int((total_sec % 3600) // 60)
        seconds = total_sec % 60.0
        print(f'Runtime     : {hours}h {minutes}m {seconds:.1f}s')
        if total_sec > 0:
            print(f'Average FPS : {final_steps / total_sec:.1f}')

    summary_cols = [
        'TotalEnvSteps',
        'Metrics/EpRet',
        'Metrics/EpCost',
        'CostRate_est',
        'Loss/Loss_pi',
    ]
    if lag_col is not None:
        summary_cols.append(lag_col)

    keep = [col for col in summary_cols if col in df.columns]
    print('\nFinal row:')
    print(df[keep].tail(1).to_string(index=False))


def plot_run(
    df: pd.DataFrame,
    meta: dict[str, Any],
    lag_col: str | None,
    out_path: Path,
    smooth_window: int,
    show: bool,
) -> None:
    """Create and save a multi-panel PPOLagAdapt diagnostic plot."""
    x = df['TotalEnvSteps']
    cost_limit = float(meta['cost_limit'])
    cost_rate_limit = cost_limit / DEFAULT_EPISODE_LEN

    schedule_name = meta['schedule_name'] or meta['exp_name'].replace('ppo_lag_adapt_', '')
    env_id = meta['env_id'] or 'unknown-env'
    seed_tag = meta['seed_tag']

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True)
    fig.suptitle(f'PPOLagAdapt [{schedule_name}]  {env_id}  {seed_tag}', fontsize=12)
    ax_ret, ax_cost, ax_rate, ax_lambda, ax_policy, ax_critic = axes.flatten()

    ax_ret.plot(x, smooth(df['Metrics/EpRet'], smooth_window), color='#1f77b4', linewidth=2)
    ax_ret.set_title('AverageEpRet')
    ax_ret.set_xlabel('TotalEnvSteps')
    ax_ret.set_ylabel('AverageEpRet')
    ax_ret.grid(True, alpha=0.3)

    ax_cost.plot(x, smooth(df['Metrics/EpCost'], smooth_window), color='#d62728', linewidth=2)
    ax_cost.axhline(cost_limit, linestyle='--', color='#444444', linewidth=1.5, label='cost limit')
    ax_cost.set_title('AverageEpCost')
    ax_cost.set_xlabel('TotalEnvSteps')
    ax_cost.set_ylabel('AverageEpCost')
    ax_cost.legend(fontsize=8)
    ax_cost.grid(True, alpha=0.3)

    ax_rate.plot(x, smooth(df['CostRate_est'], smooth_window), color='#2ca02c', linewidth=2)
    ax_rate.axhline(
        cost_rate_limit,
        linestyle='--',
        color='#444444',
        linewidth=1.5,
        label='rate limit',
    )
    ax_rate.set_title('CostRate')
    ax_rate.set_xlabel('TotalEnvSteps')
    ax_rate.set_ylabel('CostRate')
    ax_rate.legend(fontsize=8)
    ax_rate.grid(True, alpha=0.3)

    if lag_col is not None:
        ax_lambda.plot(x, smooth(df[lag_col], smooth_window), color='#9467bd', linewidth=2)
        ax_lambda.set_ylabel('Lambda')
    else:
        ax_lambda.text(0.5, 0.5, 'No lambda column found', ha='center', va='center')
    ax_lambda.set_title('LagrangeMultiplier')
    ax_lambda.set_xlabel('TotalEnvSteps')
    ax_lambda.grid(True, alpha=0.3)

    if 'Loss/Loss_pi' in df.columns:
        ax_policy.plot(x, smooth(df['Loss/Loss_pi'], smooth_window), color='#ff7f0e', linewidth=2)
    if 'Train/KL' in df.columns:
        ax_policy_kl = ax_policy.twinx()
        ax_policy_kl.plot(
            x,
            smooth(df['Train/KL'], smooth_window),
            color='#8c564b',
            linewidth=1.5,
            alpha=0.8,
        )
        ax_policy_kl.set_ylabel('Train/KL', color='#8c564b')
        ax_policy_kl.tick_params(axis='y', labelcolor='#8c564b')
    ax_policy.set_title('Policy Loss / KL')
    ax_policy.set_xlabel('TotalEnvSteps')
    ax_policy.set_ylabel('Loss/Loss_pi')
    ax_policy.grid(True, alpha=0.3)

    critic_handles = []
    if 'Loss/Loss_reward_critic' in df.columns:
        critic_handles.extend(
            ax_critic.plot(
                x,
                smooth(df['Loss/Loss_reward_critic'], smooth_window),
                color='#17becf',
                linewidth=2,
                label='reward critic',
            ),
        )
    if 'Loss/Loss_cost_critic' in df.columns:
        critic_handles.extend(
            ax_critic.plot(
                x,
                smooth(df['Loss/Loss_cost_critic'], smooth_window),
                color='#bcbd22',
                linewidth=2,
                label='cost critic',
            ),
        )
    if 'CostViolation' in df.columns:
        ax_violation = ax_critic.twinx()
        ax_violation.plot(
            x,
            smooth(df['CostViolation'], smooth_window),
            color='#7f7f7f',
            linewidth=1.5,
            alpha=0.8,
            label='cost violation',
        )
        ax_violation.axhline(0.0, linestyle='--', color='#7f7f7f', linewidth=1.0, alpha=0.6)
        ax_violation.set_ylabel('EpCost - limit', color='#7f7f7f')
        ax_violation.tick_params(axis='y', labelcolor='#7f7f7f')
    ax_critic.set_title('Critic Losses / Violation')
    ax_critic.set_xlabel('TotalEnvSteps')
    ax_critic.set_ylabel('Critic loss')
    if critic_handles:
        ax_critic.legend(fontsize=8, loc='upper right')
    ax_critic.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'\nPlot saved -> {out_path}')
    if show:
        plt.show()
    plt.close(fig)


def plot_comparison_layout(
    df: pd.DataFrame,
    meta: dict[str, Any],
    lag_col: str | None,
    out_path: Path,
    smooth_window: int,
    show: bool,
) -> None:
    """Create the same panel structure used by the PPO/TRPO comparison plot."""
    x = df['TotalEnvSteps']
    cost_limit = float(meta['cost_limit'])
    cost_rate_limit = cost_limit / DEFAULT_EPISODE_LEN

    ncols = 4 if lag_col is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3.4), constrained_layout=True)

    if ncols == 3:
        ax_ret, ax_cost, ax_rate = axes
        ax_lambda = None
    else:
        ax_ret, ax_cost, ax_rate, ax_lambda = axes

    ax_ret.plot(x, smooth(df['Metrics/EpRet'], smooth_window), linewidth=2)
    ax_ret.set_title('AverageEpRet')
    ax_ret.set_xlabel('TotalEnvSteps')
    ax_ret.set_ylabel('AverageEpRet')
    ax_ret.grid(True, alpha=0.3)

    ax_cost.plot(x, smooth(df['Metrics/EpCost'], smooth_window), linewidth=2)
    ax_cost.axhline(cost_limit, linestyle='--', linewidth=1.5)
    ax_cost.set_title('AverageEpCost')
    ax_cost.set_xlabel('TotalEnvSteps')
    ax_cost.set_ylabel('AverageEpCost')
    ax_cost.grid(True, alpha=0.3)

    ax_rate.plot(x, smooth(df['CostRate_est'], smooth_window), linewidth=2)
    ax_rate.axhline(cost_rate_limit, linestyle='--', linewidth=1.5)
    ax_rate.set_title('CostRate')
    ax_rate.set_xlabel('TotalEnvSteps')
    ax_rate.set_ylabel('CostRate')
    ax_rate.grid(True, alpha=0.3)

    if ax_lambda is not None and lag_col is not None:
        ax_lambda.plot(x, smooth(df[lag_col], smooth_window), linewidth=2)
        ax_lambda.set_title('LagrangeMultiplier')
        ax_lambda.set_xlabel('TotalEnvSteps')
        ax_lambda.set_ylabel('Lambda')
        ax_lambda.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Comparison plot saved -> {out_path}')
    if show:
        plt.show()
    plt.close(fig)


def build_comparison_out_path(out_path: Path) -> Path:
    """Derive the comparison plot path from the diagnostic plot path."""
    return out_path.with_name(f'{out_path.stem}_comparison{out_path.suffix}')


def main() -> None:
    """Parse arguments and plot one PPOLagAdapt run."""
    parser = argparse.ArgumentParser(
        description='Plot PPOLagAdapt training results with schedule diagnostics.',
    )
    parser.add_argument(
        '--schedule',
        type=str,
        default=None,
        help='Schedule name for auto-discovery, e.g. late_soft_adaptive',
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Path to progress.csv (auto-discovered if omitted).',
    )
    parser.add_argument(
        '--smooth',
        type=int,
        default=1,
        help='Rolling mean window for plotted curves.',
    )
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='Output PNG path.',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the plot window in addition to saving the PNG.',
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else find_latest_progress(args.schedule)
    print(f'Using CSV: {csv_path}')

    meta = load_run_metadata(csv_path)
    df = load_and_enrich(csv_path, meta['cost_limit'])
    lag_col = find_lagrange_column(df)
    print_run_summary(df, meta, lag_col)

    schedule_name = meta['schedule_name'] or meta['exp_name'].replace('ppo_lag_adapt_', '')
    env_id = meta['env_id'] or 'results'
    default_out = (
        repo_root()
        / 'plots'
        / 'PPOLag_ada'
        / (schedule_name or 'misc')
        / f'{env_id}_{meta["seed_tag"]}.png'
    )
    out_path = Path(args.out) if args.out else default_out

    plot_run(df, meta, lag_col, out_path, max(args.smooth, 1), args.show)
    comparison_out = build_comparison_out_path(out_path)
    plot_comparison_layout(df, meta, lag_col, comparison_out, max(args.smooth, 1), args.show)


if __name__ == '__main__':
    main()
