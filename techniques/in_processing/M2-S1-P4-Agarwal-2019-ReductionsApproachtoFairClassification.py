import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Callable
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.utils.validation import check_X_y, check_array
import warnings

def reductions_approach_fair_classification(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    sensitive_features: Union[np.ndarray, pd.Series],
    base_estimator: Optional[BaseEstimator] = None,
    constraint_type: str = "demographic_parity",
    reduction_method: str = "exponentiated_gradient",
    max_iter: int = 50,
    learning_rate: float = 2.0,
    eps: float = 0.01,
    grid_size: int = 10,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Implement the Reductions Approach to Fair Classification using either
    Exponentiated Gradient or Grid Search reduction methods.
    
    This approach reduces fair classification to a sequence of cost-sensitive
    classification problems, using Lagrange multipliers to handle fairness
    constraints formalized as linear inequalities on conditional moments.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training feature matrix
    y : array-like of shape (n_samples,)
        Binary target labels (0 or 1)
    sensitive_features : array-like of shape (n_samples,)
        Sensitive attribute values (e.g., protected group membership)
    base_estimator : BaseEstimator, optional
        Base classifier to use. If None, uses LogisticRegression
    constraint_type : str, default="demographic_parity"
        Type of fairness constraint ("demographic_parity" or "equalized_odds")
    reduction_method : str, default="exponentiated_gradient"
        Reduction algorithm ("exponentiated_gradient" or "grid_search")
    max_iter : int, default=50
        Maximum number of iterations for exponentiated gradient
    learning_rate : float, default=2.0
        Learning rate for Lagrange multiplier updates
    eps : float, default=0.01
        Tolerance for constraint violation
    grid_size : int, default=10
        Number of grid points for grid search method
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'best_classifier': Trained fair classifier
        - 'constraint_violation': Final constraint violation
        - 'multipliers': Final Lagrange multipliers
        - 'training_history': Training progress information
        - 'fairness_metrics': Computed fairness metrics
        - 'method_used': Reduction method used
    """
    
    # Input validation
    X, y = check_X_y(X, y)
    sensitive_features = check_array(sensitive_features, ensure_2d=False)
    
    if len(np.unique(y)) != 2:
        raise ValueError("Only binary classification is supported")
    
    if constraint_type not in ["demographic_parity", "equalized_odds"]:
        raise ValueError("constraint_type must be 'demographic_parity' or 'equalized_odds'")
    
    if reduction_method not in ["exponentiated_gradient", "grid_search"]:
        raise ValueError("reduction_method must be 'exponentiated_gradient' or 'grid_search'")
    
    # Set default base estimator
    if base_estimator is None:
        base_estimator = LogisticRegression(random_state=random_state)
    
    # Convert to numpy arrays for easier manipulation
    X = np.array(X)
    y = np.array(y)
    sensitive_features = np.array(sensitive_features)
    
    n_samples = X.shape[0]
    sensitive_groups = np.unique(sensitive_features)
    n_groups = len(sensitive_groups)
    
    if reduction_method == "exponentiated_gradient":
        result = _exponentiated_gradient_reduction(
            X, y, sensitive_features, base_estimator, constraint_type,
            max_iter, learning_rate, eps, random_state
        )
    else:
        result = _grid_search_reduction(
            X, y, sensitive_features, base_estimator, constraint_type,
            grid_size, eps, random_state
        )
    
    # Compute fairness metrics
    fairness_metrics = _compute_fairness_metrics(
        X, y, sensitive_features, result['best_classifier'], constraint_type
    )
    
    result['fairness_metrics'] = fairness_metrics
    result['method_used'] = reduction_method
    
    return result

def _exponentiated_gradient_reduction(
    X: np.ndarray,
    y: np.ndarray,
    sensitive_features: np.ndarray,
    base_estimator: BaseEstimator,
    constraint_type: str,
    max_iter: int,
    learning_rate: float,
    eps: float,
    random_state: Optional[int]
) -> Dict[str, Any]:
    """
    Implement the Exponentiated Gradient reduction algorithm.
    
    This method iteratively updates Lagrange multipliers using exponentiated
    gradient updates and solves cost-sensitive classification problems.
    """
    
    sensitive_groups = np.unique(sensitive_features)
    n_groups = len(sensitive_groups)
    
    # Initialize Lagrange multipliers
    # For demographic parity: one multiplier per group
    # For equalized odds: two multipliers per group (for y=0 and y=1)
    if constraint_type == "demographic_parity":
        n_multipliers = n_groups
        multipliers = np.ones(n_multipliers) / n_multipliers
    else:  # equalized_odds
        n_multipliers = 2 * n_groups
        multipliers = np.ones(n_multipliers) / n_multipliers
    
    best_classifier = None
    best_violation = float('inf')
    training_history = []
    
    for iteration in range(max_iter):
        # Solve cost-sensitive classification problem
        # Compute sample weights based on current multipliers
        sample_weights = _compute_sample_weights(
            y, sensitive_features, multipliers, constraint_type
        )
        
        # Train classifier with sample weights
        classifier = clone(base_estimator)
        if hasattr(classifier, 'fit') and 'sample_weight' in classifier.fit.__code__.co_varnames:
            classifier.fit(X, y, sample_weight=sample_weights)
        else:
            # If base estimator doesn't support sample weights, use bootstrap sampling
            indices = np.random.choice(
                len(X), size=len(X), p=sample_weights/sample_weights.sum()
            )
            classifier.fit(X[indices], y[indices])
        
        # Compute constraint violations
        violations = _compute_constraint_violations(
            X, y, sensitive_features, classifier, constraint_type
        )
        
        # Track best classifier
        max_violation = np.max(np.abs(violations))
        if max_violation < best_violation:
            best_violation = max_violation
            best_classifier = clone(classifier)
        
        training_history.append({
            'iteration': iteration,
            'max_violation': max_violation,
            'multipliers': multipliers.copy(),
            'violations': violations.copy()
        })
        
        # Check convergence
        if max_violation <= eps:
            break
        
        # Update multipliers using exponentiated gradient
        multipliers = multipliers * np.exp(learning_rate * violations / len(X))
        multipliers = multipliers / np.sum(multipliers)  # Normalize
    
    return {
        'best_classifier': best_classifier,
        'constraint_violation': best_violation,
        'multipliers': multipliers,
        'training_history': training_history
    }

def _grid_search_reduction(
    X: np.ndarray,
    y: np.ndarray,
    sensitive_features: np.ndarray,
    base_estimator: BaseEstimator,
    constraint_type: str,
    grid_size: int,
    eps: float,
    random_state: Optional[int]
) -> Dict[str, Any]:
    """
    Implement the Grid Search reduction algorithm.
    
    This method searches over a grid of Lagrange multiplier values
    and selects the best classifier that satisfies constraints.
    """
    
    sensitive_groups = np.unique(sensitive_features)
    n_groups = len(sensitive_groups)
    
    # Define grid of multiplier values
    if constraint_type == "demographic_parity":
        n_multipliers = n_groups
    else:  # equalized_odds
        n_multipliers = 2 * n_groups
    
    # Create grid of multiplier combinations
    grid_values = np.linspace(0.01, 1.0, grid_size)
    param_grid = [{'mult_' + str(i): grid_values for i in range(n_multipliers)}]
    grid = ParameterGrid(param_grid)
    
    best_classifier = None
    best_violation = float('inf')
    best_multipliers = None
    training_history = []
    
    for i, params in enumerate(grid):
        multipliers = np.array([params['mult_' + str(j)] for j in range(n_multipliers)])
        multipliers = multipliers / np.sum(multipliers)  # Normalize
        
        # Compute sample weights and train classifier
        sample_weights = _compute_sample_weights(
            y, sensitive_features, multipliers, constraint_type
        )
        
        classifier = clone(base_estimator)
        if hasattr(classifier, 'fit') and 'sample_weight' in classifier.fit.__code__.co_varnames:
            classifier.fit(X, y, sample_weight=sample_weights)
        else:
            indices = np.random.choice(
                len(X), size=len(X), p=sample_weights/sample_weights.sum()
            )
            classifier.fit(X[indices], y[indices])
        
        # Compute constraint violations
        violations = _compute_constraint_violations(
            X, y, sensitive_features, classifier, constraint_type
        )
        
        max_violation = np.max(np.abs(violations))
        
        if max_violation < best_violation:
            best_violation = max_violation
            best_classifier = clone(classifier)
            best_multipliers = multipliers.copy()
        
        training_history.append({
            'grid_point': i,
            'max_violation': max_violation,
            'multipliers': multipliers.copy(),
            'violations': violations.copy()
        })
    
    return {
        'best_classifier': best_classifier,
        'constraint_violation': best_violation,
        'multipliers': best_multipliers,
        'training_history': training_history
    }

def _compute_sample_weights(
    y: np.ndarray,
    sensitive_features: np.ndarray,
    multipliers: np.ndarray,
    constraint_type: str
) -> np.ndarray:
    """
    Compute sample weights based on Lagrange multipliers and constraint type.
    
    The weights are used to create a cost-sensitive classification problem
    that incorporates fairness constraints through the multipliers.
    """
    
    n_samples = len(y)
    weights = np.ones(n_samples)
    sensitive_groups = np.unique(sensitive_features)
    
    if constraint_type == "demographic_parity":
        # Weight samples to balance positive prediction rates across groups
        for i, group in enumerate(sensitive_groups):
            group_mask = sensitive_features == group
            weights[group_mask] *= (1 + multipliers[i])