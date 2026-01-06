from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_irf_bands(bands, shock_labels=None, response_labels=None):
    """Plot IRF median with 68/90 percent bands in an n x n grid."""
    q_50 = bands["q_50"]
    er_90 = bands["er_90"]
    er_10 = bands["er_10"]
    er_84 = bands["er_84"]
    er_16 = bands["er_16"]

    n, _, horizon = q_50.shape
    shock_labels = shock_labels or [f"shock_{i+1}" for i in range(n)]
    response_labels = response_labels or [f"resp_{i+1}" for i in range(n)]

    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.25, wspace=0.25)

    x_axis = np.arange(horizon)
    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            center = q_50[row, col, :]
            lower_90 = center - er_10[row, col, :]
            upper_90 = center + er_90[row, col, :]
            lower_68 = center - er_16[row, col, :]
            upper_68 = center + er_84[row, col, :]

            ax.plot(x_axis, center, color="black", linewidth=2)
            ax.fill_between(x_axis, lower_90, upper_90, color="lightgray", alpha=0.6)
            ax.fill_between(x_axis, lower_68, upper_68, color="gray", alpha=0.5)

            if row == 0:
                ax.set_title(shock_labels[col])
            if col == 0:
                ax.set_ylabel(response_labels[row])

    return fig


def plot_fevd_stack(ws, labels, drop_initial=0):
    """Plot stacked FEVD contributions for each variable."""
    n, s_times_n = ws.shape
    s = s_times_n // n
    ws_trimmed = ws[:, drop_initial * n :]
    horizon = s - drop_initial

    x = np.arange(drop_initial, drop_initial + horizon)
    fig, axes = plt.subplots(n, 1, figsize=(12, 8))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        cumulative = np.zeros_like(ws_trimmed[i, 0::n])
        for j, label in enumerate(labels):
            series = ws_trimmed[i, j::n]
            ax.fill_between(x, cumulative, cumulative + series, label=label, alpha=0.7)
            cumulative = cumulative + series

        ax.set_ylim(0, 1)
        ax.set_title(labels[i])
        ax.legend()

    return fig


def _prepare_samples(samples, max_params=None):
    data = np.asarray(samples)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1)

    if max_params is not None:
        data = data[:, :max_params]

    return data


def plot_chain_traces(samples, max_params=None):
    """Plot trace plots for chain samples (draws x params)."""
    data = _prepare_samples(samples, max_params=max_params)
    draws, params = data.shape
    if params == 0:
        raise ValueError("No parameters to plot")

    if params <= 5:
        ncols, nrows = 1, params
    else:
        ncols = 3
        nrows = math.ceil(params / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for i in range(params):
        ax = axes[i]
        ax.plot(data[:, i], alpha=0.7)
        ax.set_ylabel(f"param_{i+1}", fontsize=8)

    for j in range(params, len(axes)):
        fig.delaxes(axes[j])

    axes[min(params - 1, len(axes) - 1)].set_xlabel("Iteration", fontsize=10)
    plt.tight_layout()
    return fig


def plot_chain_densities(samples, max_params=None, bins=50):
    """Plot density estimates for chain samples (draws x params)."""
    data = _prepare_samples(samples, max_params=max_params)
    draws, params = data.shape
    if params == 0:
        raise ValueError("No parameters to plot")

    if params <= 5:
        ncols, nrows = 1, params
    else:
        ncols = 3
        nrows = math.ceil(params / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2 * nrows), sharex=False)
    axes = np.atleast_1d(axes).flatten()

    for i in range(params):
        ax = axes[i]
        series = data[:, i]
        ax.hist(series, bins=bins, density=True, alpha=0.6)
        if np.any(series != series[0]):
            kde = gaussian_kde(series)
            xs = np.linspace(series.min(), series.max(), 200)
            ax.plot(xs, kde(xs), linewidth=2, label="KDE")
        ax.set_ylabel(f"param_{i+1}", fontsize=8)
        ax.legend(fontsize=8)

    for j in range(params, len(axes)):
        fig.delaxes(axes[j])

    axes[min(params - 1, len(axes) - 1)].set_xlabel("Value", fontsize=10)
    plt.tight_layout()
    return fig
