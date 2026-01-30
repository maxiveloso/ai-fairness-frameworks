import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import warnings

def pareto_optimality_fairness_performance(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    sensitive_groups: Union[np.ndarray, pd.Series, List],
    base_model: Optional[Any] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    learning_rate: float = 0.01,
    projection_method: str = 'apstar',
    fairness_constraints: Optional[List[str]] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Implement Pareto Optimality for Fairness-Performance Trade-offs using APStar algorithm.
    
    This function implements the minimax Pareto fairness approach from Martinez et al. (2020),
    which formulates group fairness as a multi-objective optimization problem. Each sensitive
    group's risk is treated as a separate objective, and the algorithm finds Pareto optimal
    solutions that balance fairness and performance.
    
    The core APStar (Approximate Projection onto Star-convex sets) algorithm iteratively
    updates linear weighting vectors for group risks through convex optimization.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data features
    y : array-like of shape (n_samples,)
        Target values (binary classification: 0 or 1)
    sensitive_groups : array-like of shape (n_samples,)
        Group membership indicators for each sample
    base_model : estimator, optional
        Base classifier to use. If None, uses LogisticRegression
    max_iterations : int, default=100
        Maximum number of APStar iterations
    tolerance : float, default=1e-6
        Convergence tolerance for weight updates
    learning_rate : float, default=0.01
        Learning rate for weight updates
    projection_method : str, default='apstar'
        Method for projection onto star-convex sets
    fairness_constraints : list of str, optional
        Types of fairness constraints to enforce
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'pareto_weights': Final Pareto optimal weights for each group
        - 'group_risks': Risk values for each sensitive group
        - 'overall_risk': Overall model risk/loss
        - 'fairness_metrics': Dictionary of fairness metrics
        - 'convergence_history': History of weight updates
        - 'pareto_frontier': Points on the Pareto frontier
        - 'model': Trained fair model
        - 'iterations': Number of iterations to convergence
        
    Raises
    ------
    ValueError
        If inputs have incompatible shapes or invalid parameters
    """
    
    # Input validation
    if base_model is None:
        base_model = LogisticRegression(random_state=random_state)
    
    X, y = check_X_y(X, y)
    sensitive_groups = check_array(sensitive_groups, ensure_2d=False)
    
    if len(y) != len(sensitive_groups):
        raise ValueError("y and sensitive_groups must have the same length")
    
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only binary values (0, 1)")
    
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    
    # Get unique groups and initialize data structures
    unique_groups = np.unique(sensitive_groups)
    n_groups = len(unique_groups)
    n_samples = len(y)
    
    # Initialize group weights uniformly (Pareto weights)
    pareto_weights = np.ones(n_groups) / n_groups
    
    # Storage for convergence history
    convergence_history = {
        'weights': [pareto_weights.copy()],
        'group_risks': [],
        'overall_risks': []
    }
    
    # Create group masks for efficient computation
    group_masks = {}
    group_sizes = {}
    for i, group in enumerate(unique_groups):
        mask = sensitive_groups == group
        group_masks[i] = mask
        group_sizes[i] = np.sum(mask)
    
    # APStar algorithm main loop
    for iteration in range(max_iterations):
        # Compute sample weights based on current Pareto weights
        sample_weights = np.zeros(n_samples)
        
        for i, group in enumerate(unique_groups):
            mask = group_masks[i]
            if group_sizes[i] > 0:
                # Weight samples inversely proportional to group size, scaled by Pareto weight
                sample_weights[mask] = pareto_weights[i] / group_sizes[i]
        
        # Normalize sample weights
        sample_weights = sample_weights / np.sum(sample_weights) * n_samples
        
        # Train model with weighted samples
        try:
            model = base_model.__class__(**base_model.get_params())
            model.fit(X, y, sample_weight=sample_weights)
        except TypeError:
            # Fallback if model doesn't support sample weights
            warnings.warn("Base model doesn't support sample weights, using unweighted training")
            model = base_model.__class__(**base_model.get_params())
            model.fit(X, y)
        
        # Compute predictions and risks for each group
        y_pred_proba = model.predict_proba(X)[:, 1]
        group_risks = np.zeros(n_groups)
        
        for i, group in enumerate(unique_groups):
            mask = group_masks[i]
            if group_sizes[i] > 0:
                # Use cross-entropy loss as risk measure
                group_y = y[mask]
                group_pred = y_pred_proba[mask]
                # Clip predictions to avoid log(0)
                group_pred = np.clip(group_pred, 1e-15, 1 - 1e-15)
                group_risks[i] = -np.mean(group_y * np.log(group_pred) + 
                                        (1 - group_y) * np.log(1 - group_pred))
        
        # Compute overall risk
        y_pred_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
        overall_risk = -np.mean(y * np.log(y_pred_clipped) + 
                               (1 - y) * np.log(1 - y_pred_clipped))
        
        # Store history
        convergence_history['group_risks'].append(group_risks.copy())
        convergence_history['overall_risks'].append(overall_risk)
        
        # Update Pareto weights using gradient-based approach (APStar projection)
        if iteration < max_iterations - 1:
            # Compute gradients (risk differences from mean)
            mean_risk = np.mean(group_risks)
            risk_gradients = group_risks - mean_risk
            
            # Update weights with projection onto simplex
            new_weights = pareto_weights - learning_rate * risk_gradients
            
            # Project onto probability simplex (star-convex set)
            new_weights = _project_onto_simplex(new_weights)
            
            # Check convergence
            weight_change = np.linalg.norm(new_weights - pareto_weights)
            
            pareto_weights = new_weights
            convergence_history['weights'].append(pareto_weights.copy())
            
            if weight_change < tolerance:
                break
    
    # Compute final fairness metrics
    fairness_metrics = _compute_fairness_metrics(
        y, y_pred_proba, sensitive_groups, unique_groups
    )
    
    # Generate Pareto frontier points
    pareto_frontier = _generate_pareto_frontier(
        X, y, sensitive_groups, base_model, unique_groups, n_points=20
    )
    
    return {
        'pareto_weights': pareto_weights,
        'group_risks': group_risks,
        'overall_risk': overall_risk,
        'fairness_metrics': fairness_metrics,
        'convergence_history': convergence_history,
        'pareto_frontier': pareto_frontier,
        'model': model,
        'iterations': iteration + 1,
        'unique_groups': unique_groups,
        'group_sizes': group_sizes
    }


def _project_onto_simplex(weights: np.ndarray) -> np.ndarray:
    """
    Project weights onto the probability simplex using Euclidean projection.
    
    This implements the projection onto the star-convex set (probability simplex)
    as required by the APStar algorithm.
    """
    n = len(weights)
    
    # Sort weights in descending order
    sorted_weights = np.sort(weights)[::-1]
    
    # Find the projection
    cumsum = np.cumsum(sorted_weights)
    index = np.arange(1, n + 1)
    condition = sorted_weights - (cumsum - 1) / index > 0
    
    if np.any(condition):
        rho = np.max(np.where(condition)[0])
        theta = (cumsum[rho] - 1) / (rho + 1)
        projected = np.maximum(weights - theta, 0)
    else:
        projected = np.zeros_like(weights)
        projected[np.argmax(weights)] = 1.0
    
    return projected


def _compute_fairness_metrics(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    sensitive_groups: np.ndarray,
    unique_groups: np.ndarray
) -> Dict[str, float]:
    """Compute various fairness metrics for model evaluation."""
    
    y_pred = (y_pred_proba > 0.5).astype(int)
    metrics = {}
    
    # Demographic parity difference
    group_positive_rates = []
    for group in unique_groups:
        mask = sensitive_groups == group
        if np.sum(mask) > 0:
            positive_rate = np.mean(y_pred[mask])
            group_positive_rates.append(positive_rate)
    
    if len(group_positive_rates) > 1:
        metrics['demographic_parity_difference'] = np.max(group_positive_rates) - np.min(group_positive_rates)
    
    # Equalized odds difference
    group_tpr = []  # True positive rates
    group_fpr = []  # False positive rates
    
    for group in unique_groups:
        mask = sensitive_groups == group
        if np.sum(mask) > 0:
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            
            if np.sum(group_y_true) > 0:  # Has positive samples
                tpr = np.sum((group_y_true == 1) & (group_y_pred == 1)) / np.sum(group_y_true == 1)
                group_tpr.append(tpr)
            
            if np.sum(group_y_true == 0) > 0:  # Has negative samples
                fpr