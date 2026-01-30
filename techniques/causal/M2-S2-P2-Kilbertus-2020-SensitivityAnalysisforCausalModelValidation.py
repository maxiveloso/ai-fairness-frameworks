import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Callable, Tuple
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

def sensitivity_analysis_causal_model_validation(
    X: Union[np.ndarray, pd.DataFrame],
    A: Union[np.ndarray, pd.Series],
    Y: Union[np.ndarray, pd.Series],
    confounding_strength: Union[float, np.ndarray] = None,
    confounding_range: Tuple[float, float] = (-2.0, 2.0),
    num_confounding_values: int = 20,
    predictor_type: str = 'linear',
    noise_model: str = 'additive_gaussian',
    optimization_method: str = 'L-BFGS-B',
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    standardize_features: bool = True,
    random_state: Optional[int] = None
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Perform sensitivity analysis for causal model validation by computing the maximum
    difference between counterfactually fair predictors under varying confounding strength.
    
    This implementation follows Kilbertus et al. (2020) methodology for assessing how
    sensitive counterfactual fairness conclusions are to unmeasured confounding. The
    technique computes bounds on the difference between fair predictors under different
    assumptions about hidden confounding variables.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix containing observed covariates
    A : array-like of shape (n_samples,)
        Protected/sensitive attribute (binary: 0 or 1)
    Y : array-like of shape (n_samples,)
        Target variable (continuous or binary)
    confounding_strength : float or array-like, optional
        Specific confounding strength(s) to evaluate. If None, uses confounding_range
    confounding_range : tuple of float, default=(-2.0, 2.0)
        Range of confounding strengths to evaluate (gamma_min, gamma_max)
    num_confounding_values : int, default=20
        Number of confounding strength values to evaluate across the range
    predictor_type : str, default='linear'
        Type of predictor model ('linear' or 'nonlinear')
    noise_model : str, default='additive_gaussian'
        Assumed noise model for the additive noise model (ANM)
    optimization_method : str, default='L-BFGS-B'
        Optimization method for multivariate confounding case
    max_iterations : int, default=1000
        Maximum number of optimization iterations
    tolerance : float, default=1e-6
        Convergence tolerance for optimization
    standardize_features : bool, default=True
        Whether to standardize features before analysis
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'max_difference': Maximum difference between counterfactually fair predictors
        - 'confounding_strengths': Array of evaluated confounding strengths
        - 'sensitivity_curve': Sensitivity values across confounding strengths
        - 'critical_confounding': Confounding strength where difference is maximized
        - 'robustness_measure': Overall robustness measure (inverse of max difference)
        - 'bivariate_analysis': Results for bivariate confounding (if applicable)
        - 'multivariate_analysis': Results for multivariate confounding
        - 'convergence_info': Optimization convergence information
        
    Notes
    -----
    The sensitivity analysis evaluates how conclusions about counterfactual fairness
    change under different assumptions about unmeasured confounding. Higher sensitivity
    indicates that fairness conclusions are more fragile to hidden confounding.
    
    For bivariate confounding (single confounder), closed-form solutions are used.
    For multivariate confounding, automatic differentiation-based optimization is employed.
    
    The method assumes additive noise models (ANMs) where:
    U -> A, U -> Y (confounding structure)
    X -> A, X -> Y (observed relationships)
    
    References
    ----------
    Kilbertus, N., Ball, P. J., Kusner, M. J., Weller, A., & Silva, R. (2020).
    The sensitivity of counterfactual fairness to unmeasured confounding.
    In Proceedings of the 35th Conference on Uncertainty in Artificial Intelligence.
    """
    
    # Input validation
    X = np.asarray(X)
    A = np.asarray(A).flatten()
    Y = np.asarray(Y).flatten()
    
    if X.shape[0] != len(A) or X.shape[0] != len(Y):
        raise ValueError("X, A, and Y must have the same number of samples")
    
    if not np.all(np.isin(A, [0, 1])):
        raise ValueError("Protected attribute A must be binary (0 or 1)")
    
    if len(np.unique(A)) != 2:
        raise ValueError("Protected attribute A must have both classes present")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples, n_features = X.shape
    
    # Standardize features if requested
    if standardize_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Set up confounding strength values to evaluate
    if confounding_strength is not None:
        gamma_values = np.atleast_1d(confounding_strength)
    else:
        gamma_values = np.linspace(confounding_range[0], confounding_range[1], 
                                 num_confounding_values)
    
    # Initialize results storage
    sensitivity_values = np.zeros(len(gamma_values))
    convergence_info = []
    
    # Fit baseline models without confounding
    baseline_predictor = _fit_baseline_predictor(X, A, Y, predictor_type)
    
    # Compute sensitivity for each confounding strength
    for i, gamma in enumerate(gamma_values):
        try:
            if n_features == 1:
                # Bivariate case: use closed-form solution
                sensitivity_val, conv_info = _compute_bivariate_sensitivity(
                    X, A, Y, gamma, baseline_predictor, predictor_type
                )
            else:
                # Multivariate case: use optimization
                sensitivity_val, conv_info = _compute_multivariate_sensitivity(
                    X, A, Y, gamma, baseline_predictor, predictor_type,
                    optimization_method, max_iterations, tolerance
                )
            
            sensitivity_values[i] = sensitivity_val
            convergence_info.append(conv_info)
            
        except Exception as e:
            warnings.warn(f"Failed to compute sensitivity for gamma={gamma}: {str(e)}")
            sensitivity_values[i] = np.nan
            convergence_info.append({'success': False, 'message': str(e)})
    
    # Find maximum difference and critical confounding strength
    valid_indices = ~np.isnan(sensitivity_values)
    if not np.any(valid_indices):
        raise RuntimeError("Failed to compute sensitivity for any confounding strength")
    
    max_difference = np.nanmax(sensitivity_values)
    critical_idx = np.nanargmax(sensitivity_values)
    critical_confounding = gamma_values[critical_idx]
    
    # Compute robustness measure (inverse of sensitivity)
    robustness_measure = 1.0 / (1.0 + max_difference)
    
    # Perform detailed bivariate analysis if applicable
    bivariate_analysis = None
    if n_features == 1:
        bivariate_analysis = _detailed_bivariate_analysis(
            X, A, Y, gamma_values, baseline_predictor, predictor_type
        )
    
    # Perform multivariate analysis
    multivariate_analysis = _detailed_multivariate_analysis(
        X, A, Y, critical_confounding, baseline_predictor, predictor_type,
        optimization_method, max_iterations, tolerance
    )
    
    return {
        'max_difference': max_difference,
        'confounding_strengths': gamma_values,
        'sensitivity_curve': sensitivity_values,
        'critical_confounding': critical_confounding,
        'robustness_measure': robustness_measure,
        'bivariate_analysis': bivariate_analysis,
        'multivariate_analysis': multivariate_analysis,
        'convergence_info': convergence_info,
        'baseline_predictor_performance': _evaluate_predictor_performance(
            baseline_predictor, X, A, Y
        )
    }

def _fit_baseline_predictor(X: np.ndarray, A: np.ndarray, Y: np.ndarray, 
                          predictor_type: str) -> BaseEstimator:
    """Fit baseline predictor without confounding assumptions."""
    # Combine features and protected attribute
    X_combined = np.column_stack([X, A])
    
    if predictor_type == 'linear':
        predictor = LinearRegression()
    else:
        # For nonlinear, use polynomial features with linear regression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        
        predictor = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('linear', LinearRegression())
        ])
    
    predictor.fit(X_combined, Y)
    return predictor

def _compute_bivariate_sensitivity(X: np.ndarray, A: np.ndarray, Y: np.ndarray,
                                 gamma: float, baseline_predictor: BaseEstimator,
                                 predictor_type: str) -> Tuple[float, Dict]:
    """Compute sensitivity using closed-form solution for bivariate case."""
    # For bivariate case, we can derive analytical expressions
    # This is a simplified implementation of the closed-form updates
    
    n_samples = len(Y)
    
    # Estimate structural coefficients
    # Y = beta_X * X + beta_A * A + beta_U * U + epsilon_Y
    # A = alpha_X * X + alpha_U * U + epsilon_A
    
    # Fit models to estimate coefficients
    X_flat = X.flatten()
    
    # Model for A given X
    alpha_X = np.cov(A, X_flat)[0, 1] / np.var(X_flat) if np.var(X_flat) > 0 else 0
    
    # Model for Y given X, A
    X_combined = np.column_stack([X_flat, A])
    beta_coeffs = np.linalg.lstsq(X_combined, Y, rcond=None)[0]
    beta_X, beta_A = beta_coeffs[0], beta_coeffs[1]
    
    # Compute sensitivity based on confounding strength gamma
    # This represents the maximum difference in predictions under confounding
    confounding_effect = abs(gamma * beta_A * alpha_X)
    
    # Add noise variance contribution
    residuals_Y = Y - X_combined @ beta_coeffs
    noise_var = np.var(residuals_Y)
    
    sensitivity = confounding_effect + np.sqrt(noise_var) * abs(gamma)
    
    convergence_info = {
        'success': True,
        'method': 'closed_form',
        'coefficients': {'alpha_X': alpha_X, 'beta_X': beta_X, 'beta_A': beta_A},
        'noise_variance': noise_var