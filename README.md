# Bayesian VAR (BVAR) with Conjugate Priors

![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/bayesian-var)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/bayesian-var)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/bayesian-var)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/bayesian-var)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/bayesian-var?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/bayesian-var?style=social)

A Python implementation of a standard Bayesian Vector Autoregression (VAR) with conjugate Normal–Inverse–Wishart priors and Minnesota hyperparameter structure. This repository provides routines for:

- Specifying and estimating a VAR($p$) model
- Constructing Minnesota priors with five hyperparameters ($\lambda_0$ – $\lambda_5$)
- Sampling from the Normal–Inverse–Wishart posterior
- Computing Monte Carlo impulse response functions (IRFs)
- Performing forecast error variance decomposition (FEVD)

## Project Layout

- `data/raw/`: raw source data (Excel)
- `data/processed/`: cleaned/resampled CSVs
- `notebooks/`: exploratory notebooks
- `scripts/`: CLI entrypoints
- `src/bvar/`: package modules
- `outputs/`: generated model outputs
- `tests/`: pytest suite

## Data

The data used to run the model (to recreate the results) is located at:

- `data/raw/Tes-Bills Final.xlsx`

The code should work with more variables and different types of time series without major complication.

## Quickstart

Install in editable mode:
```bash
pip install -e .
```

Prepare the quarterly data:
```bash
python - <<'PY'
import pandas as pd
df = pd.read_excel('data/raw/Tes-Bills Final.xlsx')
df['Fecha'] = pd.to_datetime(df['Fecha'])
df = df.set_index('Fecha').resample('QE').mean().reset_index()
df[['DGS5','DGS1','TES 5 años']].to_csv('data/processed/tes_bills_quarterly.csv', index=False)
PY
```

Fit the model and draw posterior samples:
```bash
python scripts/bvar_fit.py \
  --data data/processed/tes_bills_quarterly.csv \
  --lags 3 \
  --draws 2000 \
  --output outputs/fit.npz
```

Compute IRFs, FEVD, and plots:
```bash
python scripts/bvar_infer.py \
  --fit outputs/fit.npz \
  --irf-horizon 35 \
  --output-dir outputs \
  --plots-dir outputs/plots
```

Run tests:
```bash
pytest
```

## Features

- **Hyperparameter optimization** via marginal likelihood
- **Posterior sampling** of coefficients and covariance draws
- **IRF & FEVD** routines with companion-form and Cholesky factorization
- Flexible handling of lag order $p$ and variable ordering

References:

**J. Jacobo, Una introducción a los métodos de máxima entropía y de inferencia bayesiana en econometría**

## Contributing

Contributions are welcome! Please open issues or submit pull requests at
https://github.com/pablo-reyes8

## License

This project is licensed under the Apache License 2.0.
