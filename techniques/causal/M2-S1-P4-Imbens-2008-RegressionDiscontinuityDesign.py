import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import warnings
from typing import Union, Optional, Dict, Any, Tuple

def regression_discontinuity_design(
    y: Union[np.ndarray, pd.Series],
    x: Union[np.ndarray, pd.Series], 
    threshold: float,
    treatment: Optional[Union[np.ndarray, pd.Series]] = None,
    bandwidth: Optional[float] = None,
    kernel: str = 'triangular',
    polynomial_order: int = 1,
    fuzzy: bool = False,
    alpha: float = 0.05,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Implements Regression Discontinuity Design (RDD) for causal inference.
    
    RDD exploits arbitrary cutoff rules to identify causal effects by comparing
    observations just above and below a threshold. The key assumption is that
    units cannot precisely manipulate the assignment variable around the cutoff.
    
    Parameters
    ----------
    y : array-like
        Outcome variable
    x : array-like  
        Running/forcing variable (assignment variable)
    threshold : float
        Cutoff threshold for treatment assignment
    treatment : array-like, optional
        Treatment indicator. If None, assumes sharp RDD where treatment = (x >= threshold)
    bandwidth : float, optional
        Bandwidth for local regression. If None, uses cross-validation
    kernel : str, default 'triangular'
        Kernel function ('triangular', 'rectangular', 'epanechnikov')
    polynomial_order : int, default 1
        Order of polynomial for local regression (1 = linear, 2 = quadratic)
    fuzzy : bool, default False
        Whether to implement Fuzzy RDD (treatment probability jumps at cutoff)
    alpha : float, default 0.05
        Significance level for confidence intervals
    cv_folds : int, default 5
        Number of folds for cross-validation bandwidth selection
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'treatment_effect': Estimated treatment effect at discontinuity
        - 'se': Standard error of treatment effect
        - 't_stat': t-statistic
        - 'p_value': p-value for treatment effect
        - 'ci_lower': Lower bound of confidence interval
        - 'ci_upper': Upper bound of confidence interval  
        - 'bandwidth': Bandwidth used
        - 'n_left': Number of observations below threshold
        - 'n_right': Number of observations above threshold
        - 'fuzzy_first_stage': First stage results (if fuzzy=True)
        - 'diagnostics': Diagnostic tests
        
    Raises
    ------
    ValueError
        If inputs have different lengths or invalid parameters
    """
    
    # Input validation
    y = np.asarray(y)
    x = np.asarray(x)
    
    if len(y) != len(x):
        raise ValueError("y and x must have the same length")
    
    if treatment is not None:
        treatment = np.asarray(treatment)
        if len(treatment) != len(y):
            raise ValueError("treatment must have the same length as y and x")
    else:
        # Sharp RDD: treatment determined by threshold
        treatment = (x >= threshold).astype(int)
    
    if kernel not in ['triangular', 'rectangular', 'epanechnikov']:
        raise ValueError("kernel must be 'triangular', 'rectangular', or 'epanechnikov'")
    
    if polynomial_order < 1:
        raise ValueError("polynomial_order must be at least 1")
    
    # Remove missing values
    valid_idx = ~(np.isnan(y) | np.isnan(x) | np.isnan(treatment))
    y, x, treatment = y[valid_idx], x[valid_idx], treatment[valid_idx]
    
    # Center running variable at threshold
    x_centered = x - threshold
    
    # Bandwidth selection via cross-validation if not provided
    if bandwidth is None:
        bandwidth = _select_bandwidth_cv(y, x_centered, treatment, cv_folds, kernel, polynomial_order)
    
    # Apply bandwidth restriction
    in_bandwidth = np.abs(x_centered) <= bandwidth
    y_bw = y[in_bandwidth]
    x_bw = x_centered[in_bandwidth]
    treatment_bw = treatment[in_bandwidth]
    
    # Split data by threshold
    left_idx = x_bw < 0
    right_idx = x_bw >= 0
    
    n_left = np.sum(left_idx)
    n_right = np.sum(right_idx)
    
    if n_left < polynomial_order + 1 or n_right < polynomial_order + 1:
        warnings.warn("Insufficient observations for polynomial regression on one or both sides")
    
    # Generate kernel weights
    weights = _generate_kernel_weights(x_bw, bandwidth, kernel)
    
    if fuzzy:
        # Fuzzy RDD: Two-stage approach
        # First stage: regress treatment on running variable
        first_stage_result = _estimate_discontinuity(
            treatment_bw, x_bw, weights, left_idx, right_idx, polynomial_order
        )
        
        # Reduced form: regress outcome on running variable  
        reduced_form_result = _estimate_discontinuity(
            y_bw, x_bw, weights, left_idx, right_idx, polynomial_order
        )
        
        # Treatment effect = Reduced form / First stage (Wald estimator)
        if abs(first_stage_result['discontinuity']) < 1e-10:
            raise ValueError("First stage discontinuity too small for Fuzzy RDD")
            
        treatment_effect = reduced_form_result['discontinuity'] / first_stage_result['discontinuity']
        
        # Delta method for standard error
        rf_coef = reduced_form_result['discontinuity']
        fs_coef = first_stage_result['discontinuity']
        rf_se = reduced_form_result['se']
        fs_se = first_stage_result['se']
        
        se = np.sqrt((rf_se/fs_coef)**2 + (rf_coef * fs_se / fs_coef**2)**2)
        
        fuzzy_first_stage = {
            'first_stage_coef': fs_coef,
            'first_stage_se': fs_se,
            'first_stage_t': fs_coef / fs_se,
            'first_stage_p': 2 * (1 - stats.t.cdf(abs(fs_coef / fs_se), n_left + n_right - 2*(polynomial_order + 1)))
        }
        
    else:
        # Sharp RDD: Direct estimation
        result = _estimate_discontinuity(y_bw, x_bw, weights, left_idx, right_idx, polynomial_order)
        treatment_effect = result['discontinuity']
        se = result['se']
        fuzzy_first_stage = None
    
    # Statistical inference
    df = n_left + n_right - 2 * (polynomial_order + 1)
    t_stat = treatment_effect / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    # Confidence interval
    t_critical = stats.t.ppf(1 - alpha/2, df)
    ci_lower = treatment_effect - t_critical * se
    ci_upper = treatment_effect + t_critical * se
    
    # Diagnostic tests
    diagnostics = _run_diagnostics(y, x, x_centered, treatment, bandwidth, kernel, polynomial_order)
    
    return {
        'treatment_effect': treatment_effect,
        'se': se,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bandwidth': bandwidth,
        'n_left': n_left,
        'n_right': n_right,
        'fuzzy_first_stage': fuzzy_first_stage,
        'diagnostics': diagnostics
    }

def _select_bandwidth_cv(y: np.ndarray, x_centered: np.ndarray, treatment: np.ndarray, 
                        cv_folds: int, kernel: str, polynomial_order: int) -> float:
    """Select optimal bandwidth using cross-validation."""
    
    # Try range of bandwidths
    x_range = np.max(np.abs(x_centered))
    bandwidths = np.linspace(x_range * 0.1, x_range * 0.8, 20)
    
    cv_scores = []
    
    for bw in bandwidths:
        scores = []
        
        # Simple cross-validation approach
        in_bw = np.abs(x_centered) <= bw
        if np.sum(in_bw) < cv_folds * 2:
            cv_scores.append(np.inf)
            continue
            
        y_bw = y[in_bw]
        x_bw = x_centered[in_bw]
        
        # Create polynomial features
        X = np.column_stack([x_bw**i for i in range(polynomial_order + 1)])
        
        try:
            # Use sklearn's cross-validation
            model = LinearRegression()
            cv_score = -np.mean(cross_val_score(model, X, y_bw, cv=min(cv_folds, len(y_bw)//2), 
                                              scoring='neg_mean_squared_error'))
            cv_scores.append(cv_score)
        except:
            cv_scores.append(np.inf)
    
    # Return bandwidth with minimum CV error
    optimal_idx = np.argmin(cv_scores)
    return bandwidths[optimal_idx]

def _generate_kernel_weights(x: np.ndarray, bandwidth: float, kernel: str) -> np.ndarray:
    """Generate kernel weights for local regression."""
    
    u = x / bandwidth
    
    if kernel == 'rectangular':
        weights = np.ones_like(u)
    elif kernel == 'triangular':
        weights = np.maximum(0, 1 - np.abs(u))
    elif kernel == 'epanechnikov':
        weights = np.maximum(0, 0.75 * (1 - u**2))
    
    return weights

def _estimate_discontinuity(y: np.ndarray, x: np.ndarray, weights: np.ndarray,
                          left_idx: np.ndarray, right_idx: np.ndarray, 
                          polynomial_order: int) -> Dict[str, float]:
    """Estimate discontinuity using local polynomial regression."""
    
    # Create design matrix with polynomial terms and treatment indicator
    n = len(y)
    treatment_indicator = (x >= 0).astype(float)
    
    # Polynomial terms
    X = np.column_stack([x**i for i in range(polynomial_order + 1)])
    
    # Interaction terms (treatment * polynomial terms)
    X_interact = np.column_stack([treatment_indicator * x**i for i in range(polynomial_order + 1)])
    
    # Full design matrix
    X_full = np.column_stack([X, X_interact])
    
    # Weighted least squares
    W = np.diag(weights)
    
    try:
        # (X'WX)^(-1) X'Wy
        XWX = X_full.T @ W @ X_full
        XWy = X_full.T @ W @ y
        
        beta = np.linalg.solve(XWX, XWy)
        
        # Discontinuity