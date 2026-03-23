"""Utilities for making paper-ready training plots from OmniSafe result folders."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f'Expected dict in {config_path}, got {type(data).__name__}')
    return data


def _load_progress(progress_path: Path) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    with progress_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f'No progress rows found in {progress_path}')

    columns = reader.fieldnames
    if columns is None:
        raise ValueError(f'No header found in {progress_path}')

    data: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
    for column in columns:
        values = [float(row[column]) for row in rows]
        data[column] = np.asarray(values, dtype=np.float64)
    return data


def _paper_axes_style(ax: Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, color='#d9d9d9', linewidth=0.8, alpha=0.8)
    ax.tick_params(axis='both', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _save_figure(fig: Figure, output_stem: Path) -> tuple[Path, Path]:
    pdf_path = output_stem.with_suffix('.pdf')
    png_path = output_stem.with_suffix('.png')
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return pdf_path, png_path


def plot_run_metrics(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, tuple[Path, Path]]:
    """Create episode-return and episode-cost plots for a single OmniSafe run folder."""
    run_path = Path(run_dir).expanduser().resolve()
    config_path = run_path / 'config.json'
    progress_path = run_path / 'progress.csv'
    if not config_path.exists():
        raise FileNotFoundError(f'Missing config file: {config_path}')
    if not progress_path.exists():
        raise FileNotFoundError(f'Missing progress file: {progress_path}')

    config = _load_config(config_path)
    progress = _load_progress(progress_path)
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else run_path / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    algo = str(config.get('algo', 'algo'))
    env_id = str(config.get('env_id', 'env'))
    cost_limit = (
        config.get('algo_cfgs', {}).get('cost_limit')
        if isinstance(config.get('algo_cfgs'), dict)
        else None
    )

    x = progress['TotalEnvSteps']
    ep_ret = progress['Metrics/EpRet']
    ep_cost = progress['Metrics/EpCost']

    label = f'{algo} on {env_id}'

    ret_fig, ret_ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ret_ax.plot(x, ep_ret, color='#1f77b4', linewidth=2.2, label=label)
    _paper_axes_style(ret_ax, 'Total Environment Steps', 'Episode Return')
    ret_ax.legend(frameon=False, fontsize=10)
    return_paths = _save_figure(ret_fig, out_dir / 'episode_return_vs_total_env_steps')

    cost_fig, cost_ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    cost_ax.plot(x, ep_cost, color='#d62728', linewidth=2.2, label=label)
    if isinstance(cost_limit, (int, float)):
        cost_ax.axhline(
            float(cost_limit),
            color='#444444',
            linestyle='--',
            linewidth=1.5,
            label=f'Cost Limit ({cost_limit:g})',
        )
    _paper_axes_style(cost_ax, 'Total Environment Steps', 'Episode Cost')
    cost_ax.legend(frameon=False, fontsize=10)
    cost_paths = _save_figure(cost_fig, out_dir / 'episode_cost_vs_total_env_steps')

    return {
        'return': return_paths,
        'cost': cost_paths,
    }


def find_latest_run(results_dir: str | Path) -> Path:
    """Find the most recent run directory containing both config.json and progress.csv."""
    base_dir = Path(results_dir).expanduser().resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f'Results directory does not exist: {base_dir}')

    candidates = [
        path
        for path in base_dir.rglob('*')
        if path.is_dir() and (path / 'config.json').exists() and (path / 'progress.csv').exists()
    ]
    if not candidates:
        raise FileNotFoundError(f'No OmniSafe run directories found under {base_dir}')
    return max(candidates, key=lambda path: path.stat().st_mtime)
