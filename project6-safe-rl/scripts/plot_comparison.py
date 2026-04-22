"""
plot_comparison.py — Side-by-side overlay: PPO-Lagrangian vs DMB-PPO vs PMAL-USC.

Reads one progress CSV per algorithm and overlays them on the same three
axes (AverageEpRet, AverageEpCost, CostRate), matching the Figure 7 style
from the OpenAI Safe Exploration paper (cdn.openai.com/safexp-short.pdf).

Auto-discovery precedence
--------------------------
  PPO-Lag      : runs/PPOLag-{env_id}/seed-*/progress.csv       (OmniSafe output)
  DMB-PPO      : runs/DMBPPOLag-{env_id}/seed-*/progress.csv    (run_dmbppo_grid output)
  PMAL-USC     : runs/PMALUSCLag-{env_id}/seed-*/progress.csv   (run_pmalusc_grid output)
  PIDShield    : runs/PPOPIDShield-{env_id}/seed-*/progress.csv (run_ppopidshield_grid output)

The --ppolag_csv / --dmbppo_csv / --pmalusc_csv / --pidshield_csv flags override auto-discovery.

Usage
-----
# Auto-discover all (run after all algorithms have finished):
    python scripts/plot_comparison.py

# Specify CSVs explicitly:
    python scripts/plot_comparison.py \\
        --ppolag_csv    "runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-.../progress.csv" \\
        --dmbppo_csv    "runs/DMBPPOLag-{SafetyPointGoal1-v0}/seed-000-.../progress.csv" \\
        --pmalusc_csv   "runs/PMALUSCLag-{SafetyPointGoal1-v0}/seed-000-.../progress.csv" \\
        --pidshield_csv "runs/PPOPIDShield-{SafetyPointGoal1-v0}/seed-000-.../progress.csv"

# Two-way comparison only (e.g., PPO-Lag vs PPO-PIDShield):
    python scripts/plot_comparison.py --skip_dmbppo --skip_pmalusc

# Override output file:
    python scripts/plot_comparison.py --out figs/comparison_pointgoal.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COST_LIMIT    = 25.0
COST_RATE_LIM = COST_LIMIT / 1000.0

STYLES = {
    'PPO-Lag':      dict(color='#2878b5', linewidth=2, linestyle='-',  label='PPO-Lagrangian'),
    'DMB-PPO':      dict(color='#e07b39', linewidth=2, linestyle='-',  label='DMB-PPO (ours)'),
    'PMAL-USC':     dict(color='#2ca02c', linewidth=2, linestyle='-',  label='PMAL-USC (ours)'),
    'PIDShield':    dict(color='#9467bd', linewidth=2, linestyle='-',  label='PPO-PIDShield (ours)'),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find(prefix: str) -> Path | None:
    paths = sorted(Path('runs').glob(f'{prefix}-*/*/progress.csv'))
    return paths[-1] if paths else None


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalise column names: OmniSafe (slash) and native (underscore)
    renames = {
        'Metrics_EpRet':  'Metrics/EpRet',
        'Metrics_EpCost': 'Metrics/EpCost',
        'Metrics_EpLen':  'Metrics/EpLen',
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})
    df['DeltaSteps'] = df['TotalEnvSteps'].diff().fillna(df['TotalEnvSteps'])
    df['CostRate_epoch'] = df['Metrics/EpCost'] / df['Metrics/EpLen']
    df['CumulativeCost'] = (df['DeltaSteps'] * df['CostRate_epoch']).cumsum()
    df['CostRate_est']   = df['CumulativeCost'] / df['TotalEnvSteps']
    return df


def _smooth(series: pd.Series, w: int = 3) -> pd.Series:
    """Light rolling mean to reduce epoch-level noise in the overlay plot."""
    return series.rolling(window=w, min_periods=1, center=True).mean()


# ── Comparison plot ───────────────────────────────────────────────────────────

def plot_comparison(
    dfs:      dict[str, pd.DataFrame],
    out_path: str,
    env_tag:  str = '',
    smooth:   int = 3,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    algo_names = ' vs '.join(STYLES[k]['label'] for k in STYLES if k in dfs)
    suptitle = algo_names
    if env_tag:
        suptitle += f'   [{env_tag}]'
    fig.suptitle(suptitle, fontsize=10, y=1.01)

    for algo, df in dfs.items():
        st = STYLES[algo]
        x  = df['TotalEnvSteps']

        axes[0].plot(x, _smooth(df['Metrics/EpRet'],  smooth), **st)
        axes[1].plot(x, _smooth(df['Metrics/EpCost'], smooth), **st)
        axes[2].plot(x, _smooth(df['CostRate_est'],   smooth), **st)

    # Cost-limit reference lines
    axes[1].axhline(COST_LIMIT,    linestyle='--', color='grey', linewidth=1.4,
                    label=f'cost_limit = {COST_LIMIT}')
    axes[2].axhline(COST_RATE_LIM, linestyle='--', color='grey', linewidth=1.4,
                    label=f'rate limit ≈ {COST_RATE_LIM:.4f}')

    titles  = ['AverageEpRet', 'AverageEpCost', 'CostRate']
    ylabels = ['AverageEpRet', 'AverageEpCost', 'CostRate']
    for ax, ttl, yl in zip(axes, titles, ylabels):
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel('TotalEnvSteps', fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Comparison plot saved → {out_path}')
    plt.show()


def _print_final_row(algo: str, df: pd.DataFrame) -> None:
    row = df.iloc[-1]
    print(f"  {algo:<12}  "
          f"ret={row['Metrics/EpRet']:7.2f}  "
          f"cost={row['Metrics/EpCost']:7.2f}  "
          f"cost_rate={row['CostRate_est']:.5f}  "
          f"steps={int(row['TotalEnvSteps'])}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Overlay PPO-Lag, DMB-PPO, and PMAL-USC training curves.'
    )
    parser.add_argument('--ppolag_csv',    default=None,
                        help='Path to PPO-Lag progress.csv (auto-discovered if omitted)')
    parser.add_argument('--dmbppo_csv',    default=None,
                        help='Path to DMB-PPO progress.csv (auto-discovered if omitted)')
    parser.add_argument('--pmalusc_csv',   default=None,
                        help='Path to PMAL-USC progress.csv (auto-discovered if omitted)')
    parser.add_argument('--pidshield_csv', default=None,
                        help='Path to PPO-PIDShield progress.csv (auto-discovered if omitted)')
    parser.add_argument('--skip_ppolag',    action='store_true',
                        help='Exclude PPO-Lag from the comparison plot')
    parser.add_argument('--skip_dmbppo',    action='store_true',
                        help='Exclude DMB-PPO from the comparison plot')
    parser.add_argument('--skip_pmalusc',   action='store_true',
                        help='Exclude PMAL-USC from the comparison plot')
    parser.add_argument('--skip_pidshield', action='store_true',
                        help='Exclude PPO-PIDShield from the comparison plot')
    parser.add_argument('--out',    default=None, help='Output PNG path')
    parser.add_argument('--smooth', type=int, default=3,
                        help='Rolling-average window for smoothing (default 3, set 1 to disable)')
    args = parser.parse_args()

    # Resolve CSV paths
    ppolag_path    = Path(args.ppolag_csv)    if args.ppolag_csv    else _find('PPOLag')
    dmbppo_path    = Path(args.dmbppo_csv)    if args.dmbppo_csv    else _find('DMBPPOLag')
    pmalusc_path   = Path(args.pmalusc_csv)   if args.pmalusc_csv   else _find('PMALUSCLag')
    pidshield_path = Path(args.pidshield_csv) if args.pidshield_csv else _find('PPOPIDShield')

    if all(p is None for p in [ppolag_path, dmbppo_path, pmalusc_path, pidshield_path]):
        raise FileNotFoundError(
            "Could not find progress.csv for any algorithm under runs/. "
            "Run train_ppolag.py, run_dmbppo_grid.py, run_pmalusc_grid.py, and/or "
            "run_ppopidshield_grid.py first, then re-run this script."
        )

    dfs   = {}
    paths = {}

    def _try_load(algo, path, skip):
        if skip:
            print(f'{algo:<10} : SKIPPED (--skip flag)')
            return
        if path and path.exists():
            print(f'{algo:<10} CSV : {path}')
            dfs[algo]   = load(path)
            paths[algo] = path
        else:
            print(f'{algo:<10} CSV : NOT FOUND — skipping')

    _try_load('PPO-Lag',   ppolag_path,    args.skip_ppolag)
    _try_load('DMB-PPO',   dmbppo_path,    args.skip_dmbppo)
    _try_load('PMAL-USC',  pmalusc_path,   args.skip_pmalusc)
    _try_load('PIDShield', pidshield_path, args.skip_pidshield)

    if not dfs:
        raise FileNotFoundError('No valid CSVs found. Aborting.')

    print('\nFinal-epoch metrics:')
    for algo, df in dfs.items():
        _print_final_row(algo, df)

    # Infer env tag from whichever path we have
    env_tag = ''
    for p in list(paths.values()):
        for part in p.parts:
            if '-{' in part and '}' in part:
                env_tag = part.split('{', 1)[1].rstrip('}')
                break
        if env_tag:
            break

    algo_suffix = '_'.join(k.replace('-', '') for k in dfs)
    out_path = args.out or f'comparison_{env_tag or "results"}_{algo_suffix}_plot.png'.replace('/', '_')
    plot_comparison(dfs, out_path, env_tag=env_tag, smooth=args.smooth)


# Examples:
#   python scripts/plot_comparison.py                                         # all four
#   python scripts/plot_comparison.py --skip_dmbppo --skip_pmalusc           # PPO-Lag vs PIDShield
#   python scripts/plot_comparison.py --smooth 5 --out figs/four_way.png

if __name__ == '__main__':
    main()
