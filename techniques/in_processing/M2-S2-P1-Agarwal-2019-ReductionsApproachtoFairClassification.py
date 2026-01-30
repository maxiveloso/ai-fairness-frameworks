import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, Callable
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from scipy.optimize import linprog
import warnings

def reductions_approach_fair_classification(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    sensitive_features: Union[np.ndarray, pd.Series],
    base_estimator: Optional[BaseEstimator] = None,
    constraint: str = 'demographic_parity',
    algorithm: str = 'exponentiated_gradient',
    eps: float = 0.01,
    max_iter: int = 50,
    eta0: float = 2.0,
    run_linprog_step: bool = True,
    grid_size: int = 10,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Implement the Reductions Approach to Fair Classification using either 
    Exponentiated Gradient or Grid Search algorithms.
    
    This approach reduces fair classification to a sequence of cost-sensitive 
    classification problems. It finds a randomized classifier that minimizes 
    empirical error subject to fairness constraints.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training feature matrix
    y : array-like of shape (n_samples,)
        Binary target labels (0 or 1)
    sensitive_features : array-like of shape (n_samples,)
        Sensitive attribute values (e.g., gender, race)
    base_estimator : BaseEstimator, optional
        Base classifier with fit/predict methods. Default is LogisticRegression
    constraint : str, default='demographic_parity'
        Fairness constraint type. Options: 'demographic_parity', 'equalized_odds'
    algorithm : str, default='exponentiated_gradient'
        Algorithm to use. Options: 'exponentiated_gradient', 'grid_search'
    eps : float, default=0.01
        Tolerance for fairness constraint violation
    max_iter : int, default=50
        Maximum number of iterations for exponentiated gradient
    eta0 : float, default=2.0
        Initial learning rate for exponentiated gradient
    run_linprog_step : bool, default=True
        Whether to run linear programming step for final weights
    grid_size : int, default=10
        Number of points in grid search (if using grid_search algorithm)
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'classifier': Final fair classifier
        - 'weights': Weights for randomized classifier
        - 'predictors': List of base predictors
        - 'constraint_violation': Final constraint violation
        - 'training_error': Training error of fair classifier
        - 'convergence_curve': Constraint violations over iterations
        - 'fairness_metrics': Dictionary of fairness metrics
    """
    
    # Input validation
    X = np.asarray(X)
    y = np.asarray(y)
    sensitive_features = np.asarray(sensitive_features)
    
    if X.shape[0] != len(y) or X.shape[0] != len(sensitive_features):
        raise ValueError("X, y, and sensitive_features must have same number of samples")
    
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only binary values (0, 1)")
    
    if constraint not in ['demographic_parity', 'equalized_odds']:
        raise ValueError("constraint must be 'demographic_parity' or 'equalized_odds'")
    
    if algorithm not in ['exponentiated_gradient', 'grid_search']:
        raise ValueError("algorithm must be 'exponentiated_gradient' or 'grid_search'")
    
    if base_estimator is None:
        base_estimator = LogisticRegression(random_state=random_state)
    
    np.random.seed(random_state)
    
    n_samples = X.shape[0]
    sensitive_groups = np.unique(sensitive_features)
    n_groups = len(sensitive_groups)
    
    if algorithm == 'exponentiated_gradient':
        result = _exponentiated_gradient(
            X, y, sensitive_features, base_estimator, constraint,
            eps, max_iter, eta0, run_linprog_step
        )
    else:
        result = _grid_search(
            X, y, sensitive_features, base_estimator, constraint,
            eps, grid_size
        )
    
    # Calculate fairness metrics
    fairness_metrics = _calculate_fairness_metrics(
        X, y, sensitive_features, result['classifier'], constraint
    )
    
    result['fairness_metrics'] = fairness_metrics
    
    return result

def _exponentiated_gradient(X, y, sensitive_features, base_estimator, constraint,
                          eps, max_iter, eta0, run_linprog_step):
    """Implement the Exponentiated Gradient algorithm for fair classification."""
    
    n_samples = X.shape[0]
    sensitive_groups = np.unique(sensitive_features)
    n_groups = len(sensitive_groups)
    
    # Initialize constraint multipliers (Lagrange multipliers)
    if constraint == 'demographic_parity':
        # One multiplier per group difference
        n_constraints = n_groups - 1
    else:  # equalized_odds
        # Two multipliers per group difference (TPR and FPR)
        n_constraints = 2 * (n_groups - 1)
    
    lambdas = np.ones(n_constraints) / n_constraints
    predictors = []
    convergence_curve = []
    
    for t in range(max_iter):
        # Calculate cost-sensitive weights based on current multipliers
        weights = _calculate_cost_weights(
            y, sensitive_features, lambdas, constraint
        )
        
        # Train base classifier with cost-sensitive weights
        estimator = type(base_estimator)(**base_estimator.get_params())
        estimator.fit(X, y, sample_weight=weights)
        predictors.append(estimator)
        
        # Calculate constraint violations
        violations = _calculate_constraint_violations(
            X, y, sensitive_features, estimator, constraint
        )
        
        convergence_curve.append(np.max(np.abs(violations)))
        
        # Check convergence
        if np.max(np.abs(violations)) <= eps:
            break
        
        # Update multipliers using exponentiated gradient
        eta = eta0 / np.sqrt(t + 1)  # Decreasing learning rate
        lambdas = lambdas * np.exp(eta * violations)
        lambdas = lambdas / np.sum(lambdas)  # Normalize
    
    # Determine final classifier weights
    if run_linprog_step and len(predictors) > 1:
        weights = _solve_linear_program(
            X, y, sensitive_features, predictors, constraint, eps
        )
    else:
        # Use uniform weights
        weights = np.ones(len(predictors)) / len(predictors)
    
    # Create final randomized classifier
    final_classifier = _RandomizedClassifier(predictors, weights)
    
    # Calculate final metrics
    final_violations = _calculate_constraint_violations(
        X, y, sensitive_features, final_classifier, constraint
    )
    
    training_error = 1 - final_classifier.score(X, y)
    
    return {
        'classifier': final_classifier,
        'weights': weights,
        'predictors': predictors,
        'constraint_violation': np.max(np.abs(final_violations)),
        'training_error': training_error,
        'convergence_curve': convergence_curve
    }

def _grid_search(X, y, sensitive_features, base_estimator, constraint, eps, grid_size):
    """Implement Grid Search algorithm for fair classification."""
    
    # Create grid of Lagrange multipliers
    sensitive_groups = np.unique(sensitive_features)
    n_groups = len(sensitive_groups)
    
    if constraint == 'demographic_parity':
        n_constraints = n_groups - 1
    else:
        n_constraints = 2 * (n_groups - 1)
    
    # Generate grid points
    grid_points = []
    for _ in range(grid_size):
        lambdas = np.random.exponential(1.0, n_constraints)
        lambdas = lambdas / np.sum(lambdas)
        grid_points.append(lambdas)
    
    best_classifier = None
    best_error = float('inf')
    best_violation = float('inf')
    predictors = []
    
    for lambdas in grid_points:
        # Calculate cost-sensitive weights
        weights = _calculate_cost_weights(
            y, sensitive_features, lambdas, constraint
        )
        
        # Train classifier
        estimator = type(base_estimator)(**base_estimator.get_params())
        estimator.fit(X, y, sample_weight=weights)
        predictors.append(estimator)
        
        # Evaluate constraint violation
        violations = _calculate_constraint_violations(
            X, y, sensitive_features, estimator, constraint
        )
        max_violation = np.max(np.abs(violations))
        
        # Check if constraint is satisfied
        if max_violation <= eps:
            error = 1 - estimator.score(X, y)
            if error < best_error:
                best_error = error
                best_classifier = estimator
                best_violation = max_violation
    
    if best_classifier is None:
        # If no classifier satisfies constraints, pick the one with smallest violation
        best_classifier = predictors[0]
        best_violation = np.max(np.abs(_calculate_constraint_violations(
            X, y, sensitive_features, best_classifier, constraint
        )))
        best_error = 1 - best_classifier.score(X, y)
    
    return {
        'classifier': best_classifier,
        'weights': np.array([1.0]),
        'predictors': [best_classifier],
        'constraint_violation': best_violation,
        'training_error': best_error,
        'convergence_curve': [best_violation]
    }

def _calculate_cost_weights(y, sensitive_features, lambdas, constraint):
    """Calculate cost-sensitive weights based on Lagrange multipliers."""
    
    n_samples = len(y)
    weights = np.ones(n_samples)
    sensitive_groups = np.unique(sensitive_features)
    
    if constraint == 'demographic_parity':
        # Reweight based on group membership
        for i, group in enumerate(sensitive_groups[1:]):
            group_mask = (sensitive_features == group)
            baseline_mask = (sensitive_features == sensitive_groups[0])
            
            # Adjust weights to enforce demographic parity
            weights[group_mask] += lambdas[i]
            weights[baseline_mask] -= lambdas[i] * np.sum(group_mask) / np.sum(baseline_mask)
    
    else:  # equalized_odds
        # Reweight based on group and label
        for i, group in enumerate(sensitive_groups[1:]):
            group_pos_mask = (sensitive_features == group) & (y == 1)
            group_neg_mask = (sensitive_features == group) & (y == 0)
            baseline_pos_mask = (sensitive_features == sensitive_groups[0]) & (y == 1)
            baseline_neg_mask = (sensitive_features == sensitive_groups[0]) & (y == 0)
            
            # TPR constraint
            if