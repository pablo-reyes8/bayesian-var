
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bvar.design import build_var_design
from bvar.marginal import optimize_minnesota_hyperparams
from bvar.posterior import posterior_params, sample_posterior
from bvar.priors import ar1_prior_stats, minnesota_dummy_observations


def _parse_columns(value: str | None):
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_bounds(value: str | None):
    if not value:
        return None
    bounds = []
    for item in value.split(","):
        low, high = item.split(":")
        bounds.append((float(low), float(high)))
    if len(bounds) != 5:
        raise ValueError("bounds must have 5 entries: low:high,...")
    return bounds


def _chain_seeds(seed: int | None, chains: int) -> list[int | None]:
    if seed is None:
        return [None] * chains
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**32 - 1, size=chains, dtype=np.uint32).tolist()


def _load_csv(path: str, date_col: str, columns: list[str] | None):
    df = pd.read_csv(path)
    if columns:
        return df.loc[:, columns]
    if date_col in df.columns:
        df = df.drop(columns=[date_col])
    return df


def main():
    parser = argparse.ArgumentParser(description="Fit a Bayesian VAR and save posterior draws")
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--output", required=True, help="Path to output NPZ file")
    parser.add_argument("--lags", type=int, default=3, help="Number of lags")
    parser.add_argument("--draws", type=int, default=2000, help="Posterior draws")
    parser.add_argument("--chains", type=int, default=1, help="Number of chains")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--date-col", default="Fecha", help="Date column name")
    parser.add_argument(
        "--columns",
        default=None,
        help="Comma-separated list of columns to use",
    )
    parser.add_argument(
        "--opt-starts",
        type=int,
        default=10,
        help="Random starts for hyperparameter optimization",
    )
    parser.add_argument(
        "--bounds",
        default=None,
        help="Comma-separated bounds per lambda as low:high",
    )

    args = parser.parse_args()

    if args.chains < 1:
        raise ValueError("chains must be >= 1")

    columns = _parse_columns(args.columns)
    df = _load_csv(args.data, args.date_col, columns)
    if columns is None:
        columns = list(df.columns)

    data = df.to_numpy()

    Y, X = build_var_design(data, args.lags, include_intercept=True)
    delta, su, s0 = ar1_prior_stats(data, args.lags)
    y_mean = data[: 2 * args.lags, :].mean(axis=0)

    bounds = _parse_bounds(args.bounds)
    result = optimize_minnesota_hyperparams(
        Y,
        X,
        delta,
        su,
        s0,
        y_mean,
        args.lags,
        bounds=bounds,
        n_starts=args.opt_starts,
        seed=args.seed,
    )

    lambdas = result.x
    YP, XP = minnesota_dummy_observations(lambdas, delta, su, s0, y_mean, args.lags)
    Bps, Hs, Ss, vd = posterior_params(Y, X, YP, XP)

    if args.chains == 1:
        omegas, betas = sample_posterior(Bps, Hs, Ss, vd, args.draws, seed=args.seed)
        omegas_chains = None
        betas_chains = None
    else:
        chain_seeds = _chain_seeds(args.seed, args.chains)
        n = Bps.shape[1]
        k = Bps.shape[0]
        omegas_chains = np.zeros((args.chains, args.draws, n, n))
        betas_chains = np.zeros((args.chains, args.draws, n, k))
        for idx, chain_seed in enumerate(chain_seeds):
            omegas_chains[idx], betas_chains[idx] = sample_posterior(
                Bps,
                Hs,
                Ss,
                vd,
                args.draws,
                seed=chain_seed,
            )
        omegas = omegas_chains.reshape(args.chains * args.draws, n, n)
        betas = betas_chains.reshape(args.chains * args.draws, n, k)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "omegas": omegas,
        "betas": betas,
        "lags": args.lags,
        "columns": np.array(columns, dtype=object),
        "lambdas": lambdas,
        "y_mean": y_mean,
        "delta": delta,
        "su": su,
        "vd": vd,
        "draws": args.draws,
        "chains": args.chains,
        "total_draws": omegas.shape[0],
        "seed": args.seed,
    }
    if omegas_chains is not None:
        payload["omegas_chains"] = omegas_chains
    if betas_chains is not None:
        payload["betas_chains"] = betas_chains

    np.savez(output, **payload)


if __name__ == "__main__":
    main()
