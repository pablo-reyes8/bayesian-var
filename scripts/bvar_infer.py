#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bvar.fevd import fevd_from_irf
from bvar.irf import irf_mc, irf_quantiles
from bvar.plots import (
    plot_chain_densities,
    plot_chain_traces,
    plot_fevd_stack,
    plot_irf_bands,
)


def _as_labels(arr, fallback_n):
    if arr is None:
        return [f"var_{i+1}" for i in range(fallback_n)]
    return [str(x) for x in arr]


def _save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute IRF/FEVD and plots from saved chains")
    parser.add_argument("--fit", required=True, help="Path to fit NPZ file")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--irf-horizon", type=int, default=35, help="IRF horizon")
    parser.add_argument("--fevd-horizon", type=int, default=None, help="FEVD horizon")
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Directory to save plots (optional)",
    )
    parser.add_argument(
        "--max-params",
        type=int,
        default=20,
        help="Max params to plot for chains",
    )
    parser.add_argument(
        "--fevd-drop",
        type=int,
        default=0,
        help="Drop initial horizons in FEVD stack plot",
    )
    parser.add_argument(
        "--plot-format",
        default="png",
        help="Plot file extension (png, pdf, svg)",
    )

    args = parser.parse_args()

    data = np.load(args.fit, allow_pickle=True)
    omegas = data["omegas"]
    betas = data["betas"]
    columns = data.get("columns", None)

    n = omegas.shape[1]
    labels = _as_labels(columns, n)

    irf = irf_mc(omegas, betas, args.irf_horizon)
    bands = irf_quantiles(irf, args.irf_horizon)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "irf.npz",
        irf=irf,
        horizon=args.irf_horizon,
        **bands,
    )

    fevd_horizon = args.fevd_horizon or args.irf_horizon
    irf_mean = irf.mean(axis=2)
    fevd = fevd_from_irf(irf_mean, n, fevd_horizon)
    np.savez(output_dir / "fevd.npz", fevd=fevd, horizon=fevd_horizon)

    if args.plots_dir:
        plots_dir = Path(args.plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        fig = plot_irf_bands(bands, shock_labels=labels, response_labels=labels)
        _save_fig(fig, plots_dir / f"irf_bands.{args.plot_format}")

        fig = plot_fevd_stack(fevd, labels=labels, drop_initial=args.fevd_drop)
        _save_fig(fig, plots_dir / f"fevd_stack.{args.plot_format}")

        omega_flat = omegas.reshape(omegas.shape[0], -1)
        beta_flat = betas.reshape(betas.shape[0], -1)

        fig = plot_chain_traces(omega_flat, max_params=args.max_params)
        _save_fig(fig, plots_dir / f"omega_traces.{args.plot_format}")

        fig = plot_chain_densities(omega_flat, max_params=args.max_params)
        _save_fig(fig, plots_dir / f"omega_densities.{args.plot_format}")

        fig = plot_chain_traces(beta_flat, max_params=args.max_params)
        _save_fig(fig, plots_dir / f"beta_traces.{args.plot_format}")

        fig = plot_chain_densities(beta_flat, max_params=args.max_params)
        _save_fig(fig, plots_dir / f"beta_densities.{args.plot_format}")


if __name__ == "__main__":
    main()
