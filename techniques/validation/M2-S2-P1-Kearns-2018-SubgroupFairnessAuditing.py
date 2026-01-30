import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable, Tuple, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

def subgroup_fairness_auditing(
    X: np.ndarray,
    y: np.ndarray,
    sensitive_features: np.ndarray,
    fairness_constraint: str = 'fpr',
    max_iterations: int = 100,
    gamma: float = 0.01,
    algorithm: str = 'ftpl',
    oracle_model: Optional[BaseEstimator] = None,
    convergence_threshold: float = 1e-4,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform subgroup fairness auditing using game-theoretic approach.
    
    This implements the GerryFair algorithm from Kearns et al. (2018) which frames
    fairness auditing as a zero-sum game between a Learner (trying to minimize
    prediction error) and an Auditor (trying to find subgroups with unfair treatment).
    The algorithm iteratively reweights training examples to ensure fairness across
    all possible subgroups defined by the sensitive features.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Binary target variable of shape (n_samples,)
    sensitive_features : np.ndarray
        Sensitive attributes matrix of shape (n_samples, n_sensitive_features)
    fairness_constraint : str, default='fpr'
        Type of fairness constraint ('fpr' for False Positive Rate equality,
        'fnr' for False Negative Rate equality)
    max_iterations : int, default=100
        Maximum number of game iterations
    gamma : float, default=0.01
        Fairness violation tolerance threshold
    algorithm : str, default='ftpl'
        Algorithm to use ('ftpl' for Follow the Perturbed Leader,
        'fp' for Fictitious Play)
    oracle_model : BaseEstimator, optional
        Base model to use as oracle. If None, uses LogisticRegression
    convergence_threshold : float, default=1e-4
        Threshold for convergence detection
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'final_model': Trained fair classifier
        - 'fairness_violation': Final fairness violation measure
        - 'convergence_history': List of fairness violations per iteration
        - 'sample_weights': Final sample weights
        - 'iterations': Number of iterations until convergence
        - 'converged': Whether algorithm converged
        - 'accuracy': Final model accuracy
        - 'subgroup_violations': Detailed subgroup fairness violations
    """
    
    # Input validation
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(sensitive_features, np.ndarray):
        sensitive_features = np.array(sensitive_features)
        
    if X.shape[0] != y.shape[0] or X.shape[0] != sensitive_features.shape[0]:
        raise ValueError("X, y, and sensitive_features must have same number of samples")
        
    if len(np.unique(y)) != 2:
        raise ValueError("y must be binary")
        
    if fairness_constraint not in ['fpr', 'fnr']:
        raise ValueError("fairness_constraint must be 'fpr' or 'fnr'")
        
    if algorithm not in ['ftpl', 'fp']:
        raise ValueError("algorithm must be 'ftpl' or 'fp'")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    
    # Initialize oracle model
    if oracle_model is None:
        oracle_model = LogisticRegression(random_state=random_state, max_iter=1000)
    
    # Initialize sample weights uniformly
    sample_weights = np.ones(n_samples) / n_samples
    
    # Storage for convergence tracking
    convergence_history = []
    models_history = []
    
    # Game-theoretic iterations
    for iteration in range(max_iterations):
        # Learner's move: train model with current sample weights
        current_model = oracle_model.__class__(**oracle_model.get_params())
        current_model.fit(X, y, sample_weight=sample_weights * n_samples)
        models_history.append(current_model)
        
        # Get predictions
        y_pred = current_model.predict(X)
        y_pred_proba = current_model.predict_proba(X)[:, 1] if hasattr(current_model, 'predict_proba') else y_pred
        
        # Auditor's move: find worst subgroup violation
        max_violation, violating_subgroup = _find_worst_subgroup_violation(
            y, y_pred, sensitive_features, fairness_constraint
        )
        
        convergence_history.append(max_violation)
        
        # Check convergence
        if max_violation <= gamma:
            break
            
        if iteration > 0 and abs(convergence_history[-1] - convergence_history[-2]) < convergence_threshold:
            break
        
        # Update sample weights based on algorithm choice
        if algorithm == 'ftpl':
            sample_weights = _update_weights_ftpl(
                sample_weights, violating_subgroup, max_violation, iteration
            )
        else:  # fictitious play
            sample_weights = _update_weights_fp(
                sample_weights, violating_subgroup, iteration
            )
        
        # Normalize weights
        sample_weights = sample_weights / np.sum(sample_weights)
    
    # Final model (average of all models for better stability)
    if algorithm == 'fp' and len(models_history) > 1:
        final_model = _average_models(models_history)
    else:
        final_model = models_history[-1]
    
    # Final evaluation
    final_predictions = final_model.predict(X)
    final_accuracy = accuracy_score(y, final_predictions)
    final_violation, _ = _find_worst_subgroup_violation(
        y, final_predictions, sensitive_features, fairness_constraint
    )
    
    # Detailed subgroup analysis
    subgroup_violations = _analyze_all_subgroups(
        y, final_predictions, sensitive_features, fairness_constraint
    )
    
    return {
        'final_model': final_model,
        'fairness_violation': final_violation,
        'convergence_history': convergence_history,
        'sample_weights': sample_weights,
        'iterations': iteration + 1,
        'converged': max_violation <= gamma,
        'accuracy': final_accuracy,
        'subgroup_violations': subgroup_violations
    }


def _find_worst_subgroup_violation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    constraint_type: str
) -> Tuple[float, np.ndarray]:
    """Find the subgroup with maximum fairness violation."""
    
    n_samples, n_features = sensitive_features.shape
    max_violation = 0.0
    worst_subgroup = np.zeros(n_samples, dtype=bool)
    
    # Calculate overall rate for comparison
    if constraint_type == 'fpr':
        # False Positive Rate: FP / (FP + TN)
        overall_rate = np.sum((y_pred == 1) & (y_true == 0)) / max(1, np.sum(y_true == 0))
    else:  # fnr
        # False Negative Rate: FN / (FN + TP)
        overall_rate = np.sum((y_pred == 0) & (y_true == 1)) / max(1, np.sum(y_true == 1))
    
    # Use linear regression heuristic to find violating subgroups
    # This is a simplified version - full implementation would use more sophisticated search
    for feature_idx in range(n_features):
        feature_values = sensitive_features[:, feature_idx]
        unique_values = np.unique(feature_values)
        
        for threshold in unique_values:
            # Create subgroup based on threshold
            subgroup_mask = feature_values >= threshold
            
            if np.sum(subgroup_mask) < 10:  # Skip very small subgroups
                continue
                
            # Calculate rate for this subgroup
            subgroup_y_true = y_true[subgroup_mask]
            subgroup_y_pred = y_pred[subgroup_mask]
            
            if constraint_type == 'fpr':
                subgroup_negatives = np.sum(subgroup_y_true == 0)
                if subgroup_negatives > 0:
                    subgroup_rate = np.sum((subgroup_y_pred == 1) & (subgroup_y_true == 0)) / subgroup_negatives
                else:
                    continue
            else:  # fnr
                subgroup_positives = np.sum(subgroup_y_true == 1)
                if subgroup_positives > 0:
                    subgroup_rate = np.sum((subgroup_y_pred == 0) & (subgroup_y_true == 1)) / subgroup_positives
                else:
                    continue
            
            # Calculate violation
            violation = abs(subgroup_rate - overall_rate)
            
            if violation > max_violation:
                max_violation = violation
                worst_subgroup = subgroup_mask.copy()
    
    return max_violation, worst_subgroup


def _update_weights_ftpl(
    weights: np.ndarray,
    violating_subgroup: np.ndarray,
    violation: float,
    iteration: int
) -> np.ndarray:
    """Update weights using Follow the Perturbed Leader algorithm."""
    
    # Add perturbation for exploration
    perturbation = np.random.exponential(1.0 / (iteration + 1), size=len(weights))
    
    # Increase weights for violating subgroup
    new_weights = weights.copy()
    learning_rate = 1.0 / np.sqrt(iteration + 1)
    
    new_weights[violating_subgroup] *= (1 + learning_rate * violation)
    new_weights += perturbation * learning_rate
    
    return new_weights


def _update_weights_fp(
    weights: np.ndarray,
    violating_subgroup: np.ndarray,
    iteration: int
) -> np.ndarray:
    """Update weights using Fictitious Play algorithm."""
    
    # Simple averaging approach for fictitious play
    new_weights = weights.copy()
    
    # Increase focus on violating subgroup
    subgroup_weight = 1.0 / (iteration + 1)
    new_weights = (1 - subgroup_weight) * new_weights
    new_weights[violating_subgroup] += subgroup_weight / max(1, np.sum(violating_subgroup))
    
    return new_weights


def _average_models(models: list) -> BaseEstimator:
    """Average multiple models for fictitious play."""
    # For simplicity, return the last model
    # Full implementation would properly average model parameters
    return models[-1]


def _analyze_all_subgroups(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    constraint_type: str
) -> Dict[