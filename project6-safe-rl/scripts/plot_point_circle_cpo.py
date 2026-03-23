"""Create paper-ready plots for the latest Point Circle CPO run."""

from __future__ import annotations

from pathlib import Path

from plot import find_latest_run, plot_run_metrics


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / 'results' / 'point_circle_cpo'
    output_dir = repo_root / 'plots' / 'cpo'
    run_dir = find_latest_run(results_dir)
    outputs = plot_run_metrics(run_dir, output_dir=output_dir)

    print(f'RunDir={run_dir}')
    print(f'OutputDir={output_dir}')
    print(f'ReturnPlotPDF={outputs["return"][0]}')
    print(f'ReturnPlotPNG={outputs["return"][1]}')
    print(f'CostPlotPDF={outputs["cost"][0]}')
    print(f'CostPlotPNG={outputs["cost"][1]}')


if __name__ == '__main__':
    main()
