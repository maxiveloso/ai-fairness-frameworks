import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, Tuple
from scipy import stats
import warnings

def instrumental_variables(y: Union[np.ndarray, pd.Series], 
                         X: Union[np.ndarray, pd.DataFrame],
                         Z: Union[np.ndarray, pd.DataFrame],
                         add_constant: bool = True) -> Dict[str, Any]:
    """
    Perform Two-Stage Least Squares (2SLS) instrumental variables estimation.
    
    The 2SLS estimator addresses endogeneity bias when explanatory variables are
    correlated with the error term. It uses instrumental variables that are:
    1. Relevant: correlated with the endogenous explanatory variables
    2. Exogenous: uncorrelated with the error term (exclusion restriction)
    
    The 2SLS formula is: β₂SLS = (X'PzX)⁻¹X'PzY
    where Pz = Z(Z'Z)⁻¹Z' is the projection matrix onto the instrument space.
    
    Parameters
    ----------
    y : array-like, shape (n_samples,)
        Dependent variable (outcome)
    X : array-like, shape (n_samples, n_features)
        Endogenous explanatory variables (treatment variables)
    Z : array-like, shape (n_samples, n_instruments)
        Instrumental variables (instruments)
    add_constant : bool, default=True
        Whether to add a constant term to the regression
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'coefficients': 2SLS coefficient estimates
        - 'std_errors': Standard errors of coefficients
        - 't_statistics': t-statistics for each coefficient
        - 'p_values': p-values for coefficient tests
        - 'confidence_intervals': 95% confidence intervals
        - 'first_stage_f_stat': First-stage F-statistic for instrument relevance
        - 'first_stage_p_value': p-value for first-stage F-test
        - 'sargan_test': Sargan test statistic for overidentification
        - 'sargan_p_value': p-value for Sargan test
        - 'n_observations': Number of observations
        - 'n_instruments': Number of instruments
        - 'fitted_values': Fitted values from second stage
        - 'residuals': Residuals from second stage
        
    Raises
    ------
    ValueError
        If dimensions don't match or insufficient instruments
    LinAlgError
        If matrices are singular
        
    Notes
    -----
    The two-stage procedure:
    Stage 1: Regress each endogenous variable on all instruments
    Stage 2: Regress outcome on fitted values from stage 1
    
    For identification, need at least as many instruments as endogenous variables.
    The Sargan test is only computed when overidentified (more instruments than
    endogenous variables).
    
    References
    ----------
    Angrist, J. D., & Pischke, J. S. (2008). Mostly harmless econometrics: 
    An empiricist's companion. Princeton University Press.
    """
    
    # Input validation and conversion
    y = np.asarray(y).flatten()
    X = np.asarray(X)
    Z = np.asarray(Z)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
        
    n = len(y)
    if X.shape[0] != n or Z.shape[0] != n:
        raise ValueError("All input arrays must have the same number of observations")
    
    k_endog = X.shape[1]  # Number of endogenous variables
    k_instr = Z.shape[1]  # Number of instruments
    
    if k_instr < k_endog:
        raise ValueError(f"Need at least {k_endog} instruments for {k_endog} endogenous variables")
    
    # Add constant if requested
    if add_constant:
        X = np.column_stack([np.ones(n), X])
        Z = np.column_stack([np.ones(n), Z])
        k_endog += 1
        k_instr += 1
    
    try:
        # Stage 1: Regress each endogenous variable on all instruments
        # X_hat = Z * (Z'Z)^(-1) * Z' * X = P_Z * X
        ZTZ_inv = np.linalg.inv(Z.T @ Z)
        P_Z = Z @ ZTZ_inv @ Z.T  # Projection matrix onto instrument space
        X_fitted = P_Z @ X
        
        # First-stage F-test for instrument relevance
        # Test if instruments jointly explain endogenous variables
        if add_constant:
            # F-test excludes constant
            X_endog_only = X[:, 1:]  # Exclude constant
            Z_endog_only = Z[:, 1:]  # Exclude constant
            X_fitted_endog = X_fitted[:, 1:]
            
            if k_endog > 1 and Z_endog_only.shape[1] > 0:
                # Compute F-statistic for joint significance
                rss_restricted = np.sum((X_endog_only - np.mean(X_endog_only, axis=0))**2)
                rss_unrestricted = np.sum((X_endog_only - X_fitted_endog)**2)
                f_stat = ((rss_restricted - rss_unrestricted) / (k_instr - 1)) / (rss_unrestricted / (n - k_instr))
                f_p_value = 1 - stats.f.cdf(f_stat, k_instr - 1, n - k_instr)
            else:
                f_stat = np.nan
                f_p_value = np.nan
        else:
            rss_restricted = np.sum(X**2)
            rss_unrestricted = np.sum((X - X_fitted)**2)
            f_stat = ((rss_restricted - rss_unrestricted) / k_instr) / (rss_unrestricted / (n - k_instr))
            f_p_value = 1 - stats.f.cdf(f_stat, k_instr, n - k_instr)
        
        # Stage 2: Regress y on fitted values from stage 1
        # β₂SLS = (X_fitted' * X_fitted)^(-1) * X_fitted' * y
        XTX_inv = np.linalg.inv(X_fitted.T @ X_fitted)
        beta_2sls = XTX_inv @ X_fitted.T @ y
        
        # Calculate fitted values and residuals
        y_fitted = X_fitted @ beta_2sls
        residuals = y - y_fitted
        
        # Calculate standard errors
        # Var(β₂SLS) = σ² * (X'P_Z X)^(-1)
        sigma_squared = np.sum(residuals**2) / (n - k_endog)
        var_beta = sigma_squared * XTX_inv
        std_errors = np.sqrt(np.diag(var_beta))
        
        # t-statistics and p-values
        t_stats = beta_2sls / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k_endog))
        
        # 95% confidence intervals
        t_critical = stats.t.ppf(0.975, n - k_endog)
        ci_lower = beta_2sls - t_critical * std_errors
        ci_upper = beta_2sls + t_critical * std_errors
        confidence_intervals = list(zip(ci_lower, ci_upper))
        
        # Sargan test for overidentification (only if overidentified)
        sargan_stat = np.nan
        sargan_p_value = np.nan
        
        if k_instr > k_endog:
            # Sargan test: n * R² from regression of 2SLS residuals on instruments
            Z_resid_coef = np.linalg.inv(Z.T @ Z) @ Z.T @ residuals
            fitted_residuals = Z @ Z_resid_coef
            r_squared = 1 - np.sum(residuals**2) / np.sum((residuals - np.mean(residuals))**2)
            sargan_stat = n * r_squared
            sargan_p_value = 1 - stats.chi2.cdf(sargan_stat, k_instr - k_endog)
        
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(f"Singular matrix encountered in 2SLS estimation: {e}")
    
    return {
        'coefficients': beta_2sls,
        'std_errors': std_errors,
        't_statistics': t_stats,
        'p_values': p_values,
        'confidence_intervals': confidence_intervals,
        'first_stage_f_stat': f_stat,
        'first_stage_p_value': f_p_value,
        'sargan_test': sargan_stat,
        'sargan_p_value': sargan_p_value,
        'n_observations': n,
        'n_instruments': k_instr - (1 if add_constant else 0),
        'fitted_values': y_fitted,
        'residuals': residuals
    }

if __name__ == "__main__":
    # Example: Returns to education with ability bias
    # Simulate data where education is endogenous due to unobserved ability
    np.random.seed(42)
    n = 1000
    
    # Instruments: distance to college, family income
    distance_to_college = np.random.normal(50, 20, n)  # miles
    family_income = np.random.lognormal(10, 0.5, n)   # family income
    
    # Unobserved ability affects both education and wages
    ability = np.random.normal(0, 1, n)
    
    # Education equation (endogenous)
    # Education increases with ability, family income, decreases with distance
    education = (12 + 0.5 * ability + 0.0001 * family_income - 0.02 * distance_to_college + 
                np.random.normal(0, 2, n))
    education = np.maximum(education, 8)  # Minimum 8 years
    
    # Wage equation
    # True return to education is 0.08 (8% per year)
    # Ability directly affects wages (creates endogeneity bias)
    log_wages = (1.5 + 0.08 * education + 0.3 * ability + 
                np.random.normal(0, 0.3, n))
    
    # Prepare data
    y = log_wages
    X = education  # Endogenous variable
    Z = np.column_stack([distance_to_college, family_income])  # Instruments
    
    # Run 2SLS estimation
    print("=== Instrumental Variables (2SLS) Estimation ===")
    print("Example: Returns to Education")
    print(f"Sample size: {n}")
    print(f"True return to education: 0.08")
    print()
    
    results = instrumental_variables(y, X, Z)
    
    print("Results:")
    print(f"Constant: {results['coefficients'][0]:.4f} ({results['std_errors'][0]:.4f})")
    print(f"Education: {results['coefficients'][1]:.4f} ({results['std_errors