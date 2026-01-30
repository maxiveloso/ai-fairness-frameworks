import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, Tuple
from scipy.optimize import minimize
from sklearn.base import BaseClassifier, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings

def fairness_constrained_logistic_regression(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    sensitive_features: Union[np.ndarray, pd.Series],
    fairness_constraint: str = 'demographic_parity',
    constraint_tolerance: float = 0.05,
    regularization: float = 0.01,
    max_iterations: int = 1000,
    learning_rate: float = 0.01,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Implement fairness-constrained logistic regression using covariance constraints.
    
    This function implements the approach from Zafar et al. (2017) which minimizes
    logistic loss subject to fairness constraints. The method constrains the covariance
    between sensitive attributes and the signed distance from decision boundary.
    
    Mathematical formulation:
    minimize: -∑log p(yi|xi,θ) + λ||θ||²
    subject to: |cov(sensitive_attr, decision_boundary_distance)| ≤ tolerance
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training feature matrix
    y : array-like of shape (n_samples,)
        Binary target variable (0 or 1)
    sensitive_features : array-like of shape (n_samples,)
        Binary sensitive attribute (0 or 1) for fairness constraint
    fairness_constraint : str, default='demographic_parity'
        Type of fairness constraint ('demographic_parity' or 'equalized_odds')
    constraint_tolerance : float, default=0.05
        Maximum allowed covariance between sensitive features and decision boundary
    regularization : float, default=0.01
        L2 regularization parameter
    max_iterations : int, default=1000
        Maximum number of optimization iterations
    learning_rate : float, default=0.01
        Learning rate for gradient-based optimization
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'coefficients': fitted model coefficients
        - 'intercept': fitted model intercept
        - 'fairness_violation': final covariance constraint violation
        - 'log_loss': final logistic loss value
        - 'convergence_status': whether optimization converged
        - 'n_iterations': number of iterations used
        - 'demographic_parity_difference': difference in positive prediction rates
        - 'equalized_odds_difference': difference in TPR between groups
        - 'predictions': model predictions on training data
        - 'decision_function': signed distance from decision boundary
    """
    
    # Input validation
    X, y = check_X_y(X, y, accept_sparse=False)
    sensitive_features = check_array(sensitive_features, ensure_2d=False)
    
    if len(np.unique(y)) != 2:
        raise ValueError("Target variable must be binary")
    if len(np.unique(sensitive_features)) != 2:
        raise ValueError("Sensitive features must be binary")
    if X.shape[0] != len(sensitive_features):
        raise ValueError("X and sensitive_features must have same number of samples")
    
    # Convert to binary 0/1 encoding
    y_unique = np.unique(y)
    y_binary = (y == y_unique[1]).astype(int)
    
    sens_unique = np.unique(sensitive_features)
    sens_binary = (sensitive_features == sens_unique[1]).astype(int)
    
    n_samples, n_features = X.shape
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Standardize features for numerical stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize parameters
    theta = np.random.normal(0, 0.01, n_features + 1)  # +1 for intercept
    
    def sigmoid(z):
        """Numerically stable sigmoid function"""
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def compute_predictions_and_distances(theta_params):
        """Compute predictions and signed distances from decision boundary"""
        w = theta_params[:-1]
        b = theta_params[-1]
        
        # Linear combination: Xw + b
        linear_combo = X_scaled @ w + b
        
        # Predictions (probabilities)
        probs = sigmoid(linear_combo)
        
        # Signed distance from decision boundary (normalized by ||w||)
        w_norm = np.linalg.norm(w)
        if w_norm > 1e-8:
            distances = linear_combo / w_norm
        else:
            distances = linear_combo
            
        return probs, distances, linear_combo
    
    def logistic_loss(theta_params):
        """Compute logistic loss with L2 regularization"""
        probs, _, _ = compute_predictions_and_distances(theta_params)
        
        # Clip probabilities to prevent log(0)
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        
        # Logistic loss
        loss = -np.mean(y_binary * np.log(probs) + (1 - y_binary) * np.log(1 - probs))
        
        # L2 regularization
        reg_term = regularization * np.sum(theta_params[:-1] ** 2)  # Don't regularize intercept
        
        return loss + reg_term
    
    def fairness_constraint_violation(theta_params):
        """Compute covariance constraint violation"""
        _, distances, _ = compute_predictions_and_distances(theta_params)
        
        # Compute covariance between sensitive attribute and decision boundary distance
        cov_matrix = np.cov(sens_binary, distances)
        covariance = cov_matrix[0, 1] if cov_matrix.shape == (2, 2) else 0.0
        
        return abs(covariance)
    
    def objective_function(theta_params):
        """Combined objective: loss + penalty for constraint violation"""
        loss = logistic_loss(theta_params)
        constraint_viol = fairness_constraint_violation(theta_params)
        
        # Penalty method: add large penalty if constraint is violated
        penalty = 1000.0 * max(0, constraint_viol - constraint_tolerance) ** 2
        
        return loss + penalty
    
    # Optimization using scipy.optimize.minimize
    try:
        result = minimize(
            objective_function,
            theta,
            method='BFGS',
            options={'maxiter': max_iterations, 'gtol': 1e-6}
        )
        
        optimal_theta = result.x
        converged = result.success
        n_iter = result.nit if hasattr(result, 'nit') else max_iterations
        
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}. Using unconstrained logistic regression.")
        # Fallback to standard logistic regression
        lr = LogisticRegression(C=1/regularization, random_state=random_state)
        lr.fit(X_scaled, y_binary)
        optimal_theta = np.concatenate([lr.coef_[0], [lr.intercept_[0]]])
        converged = False
        n_iter = max_iterations
    
    # Compute final metrics
    final_probs, final_distances, final_linear = compute_predictions_and_distances(optimal_theta)
    final_predictions = (final_probs > 0.5).astype(int)
    
    # Fairness metrics
    group_0_mask = sens_binary == 0
    group_1_mask = sens_binary == 1
    
    # Demographic parity difference
    if np.sum(group_0_mask) > 0 and np.sum(group_1_mask) > 0:
        dp_group_0 = np.mean(final_predictions[group_0_mask])
        dp_group_1 = np.mean(final_predictions[group_1_mask])
        demographic_parity_diff = abs(dp_group_1 - dp_group_0)
    else:
        demographic_parity_diff = 0.0
    
    # Equalized odds difference (TPR difference)
    pos_mask = y_binary == 1
    if np.sum(pos_mask & group_0_mask) > 0 and np.sum(pos_mask & group_1_mask) > 0:
        tpr_group_0 = np.mean(final_predictions[pos_mask & group_0_mask])
        tpr_group_1 = np.mean(final_predictions[pos_mask & group_1_mask])
        equalized_odds_diff = abs(tpr_group_1 - tpr_group_0)
    else:
        equalized_odds_diff = 0.0
    
    # Transform coefficients back to original scale
    original_coef = optimal_theta[:-1] / scaler.scale_
    original_intercept = optimal_theta[-1] - np.sum(original_coef * scaler.mean_)
    
    return {
        'coefficients': original_coef,
        'intercept': original_intercept,
        'fairness_violation': fairness_constraint_violation(optimal_theta),
        'log_loss': logistic_loss(optimal_theta),
        'convergence_status': converged,
        'n_iterations': n_iter,
        'demographic_parity_difference': demographic_parity_diff,
        'equalized_odds_difference': equalized_odds_diff,
        'predictions': final_predictions,
        'decision_function': final_distances,
        'constraint_tolerance': constraint_tolerance,
        'regularization': regularization
    }

if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic dataset
    n_samples = 1000
    n_features = 5
    
    # Create features
    X = np.random.randn(n_samples, n_features)
    
    # Create sensitive attribute (e.g., gender: 0=female, 1=male)
    sensitive_attr = np.random.binomial(1, 0.5, n_samples)
    
    # Create target with bias toward sensitive attribute
    # This creates unfairness in the data
    true_coef = np.array([0.5, -0.3, 0.8, -0.2, 0.4])
    bias_effect = 1.5 * sensitive_attr  # Bias toward sensitive group
    
    linear_combo = X @ true_coef + bias_effect + np.random.normal(0, 0.5, n_samples)
    y = (linear_combo > 0).astype(int)
    
    print("Fairness-Constrained Logistic Regression Example")
    print("=" * 50)
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Target distribution: {np.mean(y):.3f}")
    print(f"Sensitive attribute distribution: {np.mean(sensitive_attr):.3f}")
    
    # Fit standard logistic regression for comparison
    print("\nStandard Logistic Regression:")