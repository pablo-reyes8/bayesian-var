# Bayesian VAR (BVAR) with Conjugate Priors

A Python implementation of a standard Bayesian Vector Autoregression (VAR) with conjugate Normal–Inverse–Wishart priors and Minnesota hyperparameter structure. This repository provides routines for:

- Specifying and estimating a VAR($p$) model  
- Constructing Minnesota priors with five hyperparameters ($\lambda_0$ – $\lambda_5$)
- Sampling from the Normal–Inverse–Wishart posterior  
- Computing Monte Carlo impulse response functions (IRFs)  
- Performing forecast error variance decomposition (FEVD)  

## Data:  
The data used to run the model (to recreate the results) is located in the repository under the name *Tes-Bills Final.xlsx*, the code should work with more variables and different types of time series without major complication 

## Features

- **Hyperparameter optimization** via closed-form marginal likelihood  
- **Posterior sampling** of coefficients and covariance draws  
- **IRF & FEVD** routines with companion-form and Cholesky factorization  
- Flexible handling of lag order $p$ and variable ordering  
- Example scripts analyzing shock transmission between U.S. 5-year and 1-year T-Bills and Colombian 5-years TES

## Dependencies

- Python 3.7+  
- pandas  
- numpy  
- scipy  
- matplotlib  

Install via:
```bash
pip install pandas numpy scipy matplotlib
```

References: 

**J. Jacobo, Una introducción a los métodos de máxima entropía y de inferencia bayesiana en econometría**

## Contributing

Contributions are welcome! Please open issues or submit pull requests at  
https://github.com/pablo-reyes8

## License

This project is licensed under the Apache License 2.0.  
