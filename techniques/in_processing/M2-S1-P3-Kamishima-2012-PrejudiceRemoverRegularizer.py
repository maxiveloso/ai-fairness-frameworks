import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from typing import Union, Dict, Optional, Tuple
import warnings

def prejudice_remover_regularizer(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    sensitive_attr: Union[np.ndarray, pd.Series],
    eta: float = 1.0,
    lambda_reg: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
    random_state: Optional[int] = None
) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Implement Prejudice Remover Regularizer for fairness-aware classification.
    
    This technique adds a prejudice index regularization term to logistic regression
    to reduce discrimination while maintaining predictive accuracy. The prejudice
    index measures mutual information between predictions and sensitive attributes.
    
    The objective function is: -L(D;Θ) + η R(D, Θ) + (λ/2) ||Θ||₂²
    where L is log-likelihood, R is prejudice index, η controls fairness-accuracy
    tradeoff, and λ is L2 regularization parameter.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Binary target variable (0 or 1)
    sensitive_attr : array-like of shape (n_samples,)
        Binary sensitive attribute (0 or 1)
    eta : float, default=1.0
        Fairness regularization parameter. Higher values prioritize fairness
    lambda_reg : float, default=1.0
        L2 regularization parameter for model complexity
    max_iter : int, default=1000
        Maximum number of optimization iterations
    tol : float, default=1e-6
        Convergence tolerance
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'coefficients': Learned model coefficients
        - 'intercept': Model intercept
        - 'prejudice_index': Final prejudice index value
        - 'accuracy': Model accuracy on training data
        - 'log_likelihood': Final log-likelihood
        - 'objective_value': Final objective function value
        - 'convergence_status': Optimization convergence message
        - 'fairness_metrics': Dictionary with demographic parity and equalized odds
    """
    
    # Input validation
    X = np.asarray(X)
    y = np.asarray(y).flatten()
    sensitive_attr = np.asarray(sensitive_attr).flatten()
    
    if X.shape[0] != len(y) or X.shape[0] != len(sensitive_attr):
        raise ValueError("X, y, and sensitive_attr must have same number of samples")
    
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must be binary (0 or 1)")
    
    if not np.all(np.isin(sensitive_attr, [0, 1])):
        raise ValueError("sensitive_attr must be binary (0 or 1)")
    
    if eta < 0 or lambda_reg < 0:
        raise ValueError("eta and lambda_reg must be non-negative")
    
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_samples, n_features = X_scaled.shape
    
    def sigmoid(z):
        """Sigmoid activation function with numerical stability"""
        return expit(z)
    
    def compute_prejudice_index(y_pred_proba, sensitive_attr, y_true):
        """
        Compute prejudice index (mutual information between predictions and sensitive attribute)
        PI = Σ_{Y,S} P̂r[Y,S] ln(P̂r[Y,S]/(P̂r[S] P̂r[Y]))
        """
        # Discretize predictions into binary
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Compute joint and marginal probabilities with smoothing
        eps = 1e-8
        
        # Joint probabilities P(Y, S)
        p_y0_s0 = np.mean((y_pred == 0) & (sensitive_attr == 0)) + eps
        p_y0_s1 = np.mean((y_pred == 0) & (sensitive_attr == 1)) + eps
        p_y1_s0 = np.mean((y_pred == 1) & (sensitive_attr == 0)) + eps
        p_y1_s1 = np.mean((y_pred == 1) & (sensitive_attr == 1)) + eps
        
        # Marginal probabilities
        p_y0 = np.mean(y_pred == 0) + eps
        p_y1 = np.mean(y_pred == 1) + eps
        p_s0 = np.mean(sensitive_attr == 0) + eps
        p_s1 = np.mean(sensitive_attr == 1) + eps
        
        # Prejudice index calculation
        pi = (p_y0_s0 * np.log(p_y0_s0 / (p_y0 * p_s0)) +
              p_y0_s1 * np.log(p_y0_s1 / (p_y0 * p_s1)) +
              p_y1_s0 * np.log(p_y1_s0 / (p_y1 * p_s0)) +
              p_y1_s1 * np.log(p_y1_s1 / (p_y1 * p_s1)))
        
        return pi
    
    def objective_function(theta):
        """
        Objective function: -L(D;Θ) + η R(D, Θ) + (λ/2) ||Θ||₂²
        """
        # Split parameters
        w = theta[:-1]  # coefficients
        b = theta[-1]   # intercept
        
        # Predictions
        z = X_scaled @ w + b
        y_pred_proba = sigmoid(z)
        
        # Avoid numerical issues
        y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
        
        # Log-likelihood (negative for minimization)
        log_likelihood = -np.mean(y * np.log(y_pred_proba) + (1 - y) * np.log(1 - y_pred_proba))
        
        # Prejudice index regularization
        prejudice_reg = eta * compute_prejudice_index(y_pred_proba, sensitive_attr, y)
        
        # L2 regularization (only on weights, not intercept)
        l2_reg = (lambda_reg / 2) * np.sum(w ** 2)
        
        return log_likelihood + prejudice_reg + l2_reg
    
    def gradient(theta):
        """Compute gradient of objective function"""
        w = theta[:-1]
        b = theta[-1]
        
        z = X_scaled @ w + b
        y_pred_proba = sigmoid(z)
        y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
        
        # Gradient of log-likelihood
        error = y_pred_proba - y
        grad_w = (X_scaled.T @ error) / n_samples + lambda_reg * w
        grad_b = np.mean(error)
        
        # Note: Prejudice index gradient is complex and approximated numerically
        return np.concatenate([grad_w, [grad_b]])
    
    # Initialize parameters
    initial_theta = np.random.normal(0, 0.01, n_features + 1)
    
    # Optimize
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            objective_function,
            initial_theta,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol}
        )
    
    # Extract final parameters
    final_w = result.x[:-1]
    final_b = result.x[-1]
    
    # Final predictions and metrics
    final_z = X_scaled @ final_w + final_b
    final_proba = sigmoid(final_z)
    final_pred = (final_proba > 0.5).astype(int)
    
    # Compute final metrics
    final_accuracy = accuracy_score(y, final_pred)
    final_log_likelihood = -log_loss(y, final_proba)
    final_prejudice_index = compute_prejudice_index(final_proba, sensitive_attr, y)
    
    # Fairness metrics
    # Demographic parity: P(Ŷ=1|S=0) - P(Ŷ=1|S=1)
    demo_parity = (np.mean(final_pred[sensitive_attr == 0]) - 
                   np.mean(final_pred[sensitive_attr == 1]))
    
    # Equalized odds: TPR difference and FPR difference
    tpr_s0 = np.mean(final_pred[(y == 1) & (sensitive_attr == 0)]) if np.any((y == 1) & (sensitive_attr == 0)) else 0
    tpr_s1 = np.mean(final_pred[(y == 1) & (sensitive_attr == 1)]) if np.any((y == 1) & (sensitive_attr == 1)) else 0
    fpr_s0 = np.mean(final_pred[(y == 0) & (sensitive_attr == 0)]) if np.any((y == 0) & (sensitive_attr == 0)) else 0
    fpr_s1 = np.mean(final_pred[(y == 0) & (sensitive_attr == 1)]) if np.any((y == 0) & (sensitive_attr == 1)) else 0
    
    eq_odds_tpr = tpr_s0 - tpr_s1
    eq_odds_fpr = fpr_s0 - fpr_s1
    
    return {
        'coefficients': final_w,
        'intercept': final_b,
        'prejudice_index': final_prejudice_index,
        'accuracy': final_accuracy,
        'log_likelihood': final_log_likelihood,
        'objective_value': result.fun,
        'convergence_status': result.message,
        'fairness_metrics': {
            'demographic_parity_difference': demo_parity,
            'equalized_odds_tpr_difference': eq_odds_tpr,
            'equalized_odds_fpr_difference': eq_odds_fpr
        }
    }

if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic dataset
    n_samples = 1000
    n_features = 5
    
    # Create correlated features and sensitive attribute
    sensitive = np.random.binomial(1, 0.3, n_samples)
    X = np.random.randn(n_samples, n_features)
    
    #