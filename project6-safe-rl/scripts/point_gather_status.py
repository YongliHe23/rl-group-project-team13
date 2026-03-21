"""Summarize training progress for a Point Gather CPO run."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    """Return the repository root for project6-safe-rl."""
    return Path(__file__).resolve().parents[1]


def default_runs_root() -> Path:
    """Return the default directory where full Point Gather runs are stored."""
    return repo_root() / 'results' / 'point_gather_cpo' / 'CPO-{SafetyPointGather1-v0}'


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Show status for a Point Gather CPO training run.',
    )
    parser.add_argument(
        '--run-dir',
        type=Path,
        default=None,
        help='Specific run directory containing progress.csv and config.json.',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Render quick return/cost plots to a PNG next to progress.csv.',
    )
    parser.add_argument(
        '--watch',
        type=float,
        default=None,
        help='Refresh every N seconds until interrupted.',
    )
    return parser.parse_args()


def find_latest_run(runs_root: Path) -> Path:
    """Find the most recently modified seed directory."""
    run_dirs = [path for path in runs_root.glob('seed-*') if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f'No run directories found under {runs_root}')
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def read_config(run_dir: Path) -> dict[str, Any]:
    """Load the run config if it exists."""
    config_path = run_dir / 'config.json'
    if not config_path.exists():
        return {}
    with config_path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def read_progress_rows(progress_path: Path) -> list[dict[str, str]]:
    """Load all progress rows from the CSV if present."""
    if not progress_path.exists() or progress_path.stat().st_size == 0:
        return []
    with progress_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def as_float(row: dict[str, str], key: str) -> float | None:
    """Read a numeric value from a CSV row if present."""
    value = row.get(key, '')
    if value == '':
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_number(value: float | None, digits: int = 3) -> str:
    """Format a float or display n/a."""
    if value is None:
        return 'n/a'
    return f'{value:.{digits}f}'


def estimate_eta(total_steps: float | None, current_steps: float | None, total_time: float | None) -> str:
    """Estimate remaining time from current throughput."""
    if total_steps is None or current_steps is None or total_time is None:
        return 'n/a'
    if current_steps <= 0 or total_time <= 0 or current_steps >= total_steps:
        return 'n/a'
    steps_per_second = current_steps / total_time
    if steps_per_second <= 0:
        return 'n/a'
    remaining_seconds = (total_steps - current_steps) / steps_per_second
    hours = remaining_seconds / 3600
    return f'{hours:.2f}h'


def maybe_make_plot(run_dir: Path, rows: list[dict[str, str]]) -> Path | None:
    """Write a quick progress plot if matplotlib is available and rows exist."""
    if not rows:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    steps = [as_float(row, 'TotalEnvSteps') for row in rows]
    ep_ret = [as_float(row, 'Metrics/EpRet') for row in rows]
    ep_cost = [as_float(row, 'Metrics/EpCost') for row in rows]

    valid = [
        (step, ret, cost)
        for step, ret, cost in zip(steps, ep_ret, ep_cost)
        if step is not None and ret is not None and cost is not None
    ]
    if not valid:
        return None

    plot_path = run_dir / 'status_plot.png'
    x_vals = [item[0] for item in valid]
    ret_vals = [item[1] for item in valid]
    cost_vals = [item[2] for item in valid]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(x_vals, ret_vals, color='tab:blue')
    axes[0].set_ylabel('EpRet')
    axes[0].set_title('Point Gather CPO Progress')

    axes[1].plot(x_vals, cost_vals, color='tab:red')
    axes[1].axhline(0.1, color='black', linestyle='--', linewidth=1)
    axes[1].set_ylabel('EpCost')
    axes[1].set_xlabel('Env Steps')

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def print_status(run_dir: Path, rows: list[dict[str, str]], config: dict[str, Any], plot_path: Path | None) -> None:
    """Print a compact status summary."""
    progress_path = run_dir / 'progress.csv'
    modified = datetime.fromtimestamp(progress_path.stat().st_mtime).isoformat(timespec='seconds')
    print(f'RunDir: {run_dir}')
    print(f'ProgressCsv: {progress_path}')
    print(f'LastUpdated: {modified}')

    total_steps_cfg = config.get('train_cfgs', {}).get('total_steps')
    if rows:
        latest = rows[-1]
        epoch = as_float(latest, 'Train/Epoch')
        env_steps = as_float(latest, 'TotalEnvSteps')
        ep_ret = as_float(latest, 'Metrics/EpRet')
        ep_cost = as_float(latest, 'Metrics/EpCost')
        ep_len = as_float(latest, 'Metrics/EpLen')
        elapsed = as_float(latest, 'Time/Total')
        fps = as_float(latest, 'Time/FPS')
        eta = estimate_eta(
            float(total_steps_cfg) if isinstance(total_steps_cfg, (int, float)) else None,
            env_steps,
            elapsed,
        )

        print(f'Rows: {len(rows)}')
        print(f'LatestEpoch: {format_number(epoch, 0)}')
        print(f'TotalEnvSteps: {format_number(env_steps, 0)}')
        print(f'EpRet: {format_number(ep_ret)}')
        print(f'EpCost: {format_number(ep_cost)}')
        print(f'EpLen: {format_number(ep_len)}')
        print(f'ElapsedSeconds: {format_number(elapsed)}')
        print(f'FPS: {format_number(fps)}')
        print(f'ETA: {eta}')
    else:
        print('Rows: 0')
        print('Status: training has started but no progress rows are written yet.')

    if total_steps_cfg is not None:
        print(f'ConfiguredTotalSteps: {total_steps_cfg}')
    if plot_path is not None:
        print(f'Plot: {plot_path}')


def run_once(args: argparse.Namespace) -> None:
    """Run one status refresh."""
    run_dir = args.run_dir or find_latest_run(default_runs_root())
    progress_path = run_dir / 'progress.csv'
    config = read_config(run_dir)
    rows = read_progress_rows(progress_path)
    plot_path = maybe_make_plot(run_dir, rows) if args.plot else None
    print_status(run_dir, rows, config, plot_path)


def main() -> None:
    """Entry point."""
    args = parse_args()
    if args.watch is None:
        run_once(args)
        return

    while True:
        print('\033[2J\033[H', end='')
        run_once(args)
        time.sleep(args.watch)


if __name__ == '__main__':
    main()
