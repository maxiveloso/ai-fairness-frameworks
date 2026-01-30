import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

def subgroup_regularization_fairness_gerrymandering_prevention(
    X: np.ndarray,
    y: np.ndarray,
    protected_attributes: np.ndarray,
    base_learner: Optional[BaseEstimator] = None,
    gamma: float = 0.01,
    max_iterations: int = 100,
    auditor_type: str = 'fictitious_play',
    fairness_constraint: float = 0.05,
    subgroup_functions: Optional[List[Callable]] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Implements subgroup-specific regularization for fairness gerrymandering prevention
    using game-theoretic formulation with Learner-Auditor zero-sum game.
    
    This technique addresses fairness across exponentially many subgroups defined by
    structured functions over protected attributes. Uses either Follow the Perturbed
    Leader + best response or Fictitious Play algorithms.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Binary target variable of shape (n_samples,)
    protected_attributes : np.ndarray
        Protected attributes matrix of shape (n_samples, n_protected_features)
    base_learner : BaseEstimator, optional
        Base learning algorithm (default: LogisticRegression)
    gamma : float, default=0.01
        Learning rate for the game-theoretic updates
    max_iterations : int, default=100
        Maximum number of iterations for the game
    auditor_type : str, default='fictitious_play'
        Type of auditor algorithm ('fictitious_play' or 'perturbed_leader')
    fairness_constraint : float, default=0.05
        Maximum allowed unfairness (violation threshold)
    subgroup_functions : List[Callable], optional
        List of functions defining subgroups over protected attributes
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'final_classifier': Trained fair classifier
        - 'fairness_violations': Maximum fairness violation found
        - 'convergence_history': History of fairness violations over iterations
        - 'auditor_weights': Final auditor distribution over subgroups
        - 'learner_weights': Final learner distribution over hypotheses
        - 'game_value': Final value of the zero-sum game
        - 'converged': Whether the algorithm converged
        - 'subgroup_violations': Violations for each subgroup
    """
    
    # Input validation
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("X and y must be numpy arrays")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    
    if not isinstance(protected_attributes, np.ndarray):
        raise ValueError("protected_attributes must be numpy array")
    
    if protected_attributes.shape[0] != X.shape[0]:
        raise ValueError("protected_attributes must have same number of samples as X")
    
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must be binary (0, 1)")
    
    if gamma <= 0 or gamma > 1:
        raise ValueError("gamma must be in (0, 1]")
    
    if fairness_constraint <= 0:
        raise ValueError("fairness_constraint must be positive")
    
    if auditor_type not in ['fictitious_play', 'perturbed_leader']:
        raise ValueError("auditor_type must be 'fictitious_play' or 'perturbed_leader'")
    
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize base learner
    if base_learner is None:
        base_learner = LogisticRegression(random_state=random_state)
    
    n_samples, n_features = X.shape
    n_protected = protected_attributes.shape[1]
    
    # Define default subgroup functions if not provided
    # These create subgroups based on conjunctions of protected attribute values
    if subgroup_functions is None:
        subgroup_functions = _generate_default_subgroup_functions(protected_attributes)
    
    n_subgroups = len(subgroup_functions)
    
    # Initialize game components
    # Auditor maintains distribution over subgroups
    auditor_weights = np.ones(n_subgroups) / n_subgroups
    auditor_cumulative = np.zeros(n_subgroups)
    
    # Learner maintains distribution over hypotheses (classifiers)
    learner_hypotheses = []
    learner_weights = []
    learner_cumulative = []
    
    # Track convergence
    convergence_history = []
    fairness_violations = []
    
    for iteration in range(max_iterations):
        # Learner's best response: train classifier to minimize weighted loss
        # Weight samples according to auditor's current subgroup distribution
        sample_weights = _compute_sample_weights(protected_attributes, subgroup_functions, auditor_weights)
        
        # Train classifier with weighted samples
        current_classifier = _train_weighted_classifier(X, y, sample_weights, base_learner)
        learner_hypotheses.append(current_classifier)
        
        # Compute fairness violations for all subgroups
        subgroup_violations = _compute_subgroup_violations(
            X, y, protected_attributes, current_classifier, subgroup_functions
        )
        
        # Auditor's best response: find subgroup with maximum violation
        max_violation_idx = np.argmax(np.abs(subgroup_violations))
        max_violation = np.abs(subgroup_violations[max_violation_idx])
        
        fairness_violations.append(max_violation)
        convergence_history.append({
            'iteration': iteration,
            'max_violation': max_violation,
            'avg_violation': np.mean(np.abs(subgroup_violations)),
            'num_violated_subgroups': np.sum(np.abs(subgroup_violations) > fairness_constraint)
        })
        
        # Update strategies based on auditor type
        if auditor_type == 'fictitious_play':
            # Fictitious Play: update cumulative counts and recompute mixed strategies
            auditor_cumulative[max_violation_idx] += 1
            auditor_weights = auditor_cumulative / np.sum(auditor_cumulative)
            
            # Add current classifier to learner's strategy
            learner_cumulative.append(1.0)
            learner_weights = np.array(learner_cumulative) / np.sum(learner_cumulative)
            
        else:  # perturbed_leader
            # Follow the Perturbed Leader: add noise and select best response
            noise = np.random.exponential(1/gamma, n_subgroups)
            perturbed_violations = np.abs(subgroup_violations) + noise
            best_subgroup = np.argmax(perturbed_violations)
            
            # Update auditor weights with exponential weights
            auditor_weights *= np.exp(gamma * np.abs(subgroup_violations))
            auditor_weights /= np.sum(auditor_weights)
            
            # Learner uses uniform distribution over recent hypotheses
            recent_window = min(10, len(learner_hypotheses))
            learner_weights = np.zeros(len(learner_hypotheses))
            learner_weights[-recent_window:] = 1.0 / recent_window
        
        # Check convergence
        if max_violation <= fairness_constraint:
            converged = True
            break
    else:
        converged = False
    
    # Compute final mixed classifier
    if len(learner_weights) == 0:
        learner_weights = [1.0]
    
    final_classifier = _create_mixed_classifier(learner_hypotheses, learner_weights)
    
    # Compute final statistics
    final_violations = _compute_subgroup_violations(
        X, y, protected_attributes, final_classifier, subgroup_functions
    )
    
    game_value = np.max(np.abs(final_violations))
    
    return {
        'final_classifier': final_classifier,
        'fairness_violations': game_value,
        'convergence_history': convergence_history,
        'auditor_weights': auditor_weights,
        'learner_weights': np.array(learner_weights) if len(learner_weights) > 0 else np.array([1.0]),
        'game_value': game_value,
        'converged': converged,
        'subgroup_violations': final_violations,
        'n_subgroups': n_subgroups,
        'n_iterations': iteration + 1
    }


def _generate_default_subgroup_functions(protected_attributes: np.ndarray) -> List[Callable]:
    """Generate default subgroup functions based on protected attribute combinations."""
    n_samples, n_protected = protected_attributes.shape
    subgroup_functions = []
    
    # Individual protected attributes
    for i in range(n_protected):
        unique_vals = np.unique(protected_attributes[:, i])
        for val in unique_vals:
            def subgroup_func(pa, idx=i, value=val):
                return pa[:, idx] == value
            subgroup_functions.append(subgroup_func)
    
    # Pairwise combinations (if not too many)
    if n_protected <= 3:
        for i in range(n_protected):
            for j in range(i+1, n_protected):
                unique_i = np.unique(protected_attributes[:, i])
                unique_j = np.unique(protected_attributes[:, j])
                for val_i in unique_i:
                    for val_j in unique_j:
                        def subgroup_func(pa, idx_i=i, idx_j=j, val_i=val_i, val_j=val_j):
                            return (pa[:, idx_i] == val_i) & (pa[:, idx_j] == val_j)
                        subgroup_functions.append(subgroup_func)
    
    return subgroup_functions


def _compute_sample_weights(protected_attributes: np.ndarray, 
                          subgroup_functions: List[Callable],
                          auditor_weights: np.ndarray) -> np.ndarray:
    """Compute sample weights based on auditor's subgroup distribution."""
    n_samples = protected_attributes.shape[0]
    sample_weights = np.zeros(n_samples)
    
    for i, subgroup_func in enumerate(subgroup_functions):
        subgroup_mask = subgroup_func(protected_attributes)
        sample_weights[subgroup_mask] += auditor_weights[i]
    
    # Normalize to avoid zero weights
    sample_weights = np.maximum(sample_weights, 1e-8)
    return sample_weights


def _train_weighted_classifier(X: np.ndarray, y: np.ndarray, 
                             sample_weights: np.ndarray,
                             base_learner: BaseEstimator) -> BaseEstimator:
    """Train classifier with weighted samples."""
    from sklearn.base import clone
    classifier = clone(base_learner)
    
    # Check if classifier supports sample weights
    try:
        classifier.fit(X, y, sample_weight=sample_weights)
    except TypeError:
        # If sample weights not supported, use weighted resampling
        n_samples = len(y)