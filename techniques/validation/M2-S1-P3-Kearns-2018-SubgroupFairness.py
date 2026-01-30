import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Callable, Tuple
from sklearn.base import BaseClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

def subgroup_fairness(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    protected_attributes: Union[np.ndarray, pd.DataFrame],
    base_classifier: Optional[BaseClassifier] = None,
    fairness_constraint: str = 'fpr',
    max_iterations: int = 100,
    learning_rate: float = 0.1,
    tolerance: float = 1e-4,
    max_subgroup_size: int = 10,
    random_state: Optional[int] = None
) -> Dict[str, Union[float, np.ndarray, List[Dict]]]:
    """
    Implement subgroup fairness using a two-player zero-sum game between Learner and Auditor.
    
    This function implements the Fair ERM (Empirical Risk Minimization) approach from Kearns et al. (2018)
    that prevents fairness gerrymandering by ensuring fairness across all possible subgroups defined by
    protected attributes. The method uses Fictitious Play algorithm to solve the game between:
    - Learner: tries to minimize prediction error
    - Auditor: tries to find subgroups where fairness constraints are violated
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training features
    y : array-like of shape (n_samples,)
        Binary target variable (0 or 1)
    protected_attributes : array-like of shape (n_samples, n_protected_features)
        Protected attributes used to define subgroups
    base_classifier : BaseClassifier, optional
        Sklearn classifier to use. Default is LogisticRegression
    fairness_constraint : str, default='fpr'
        Type of fairness constraint ('fpr' for false positive rate, 'fnr' for false negative rate)
    max_iterations : int, default=100
        Maximum iterations for Fictitious Play algorithm
    learning_rate : float, default=0.1
        Learning rate for updating classifier weights
    tolerance : float, default=1e-4
        Convergence tolerance for the game
    max_subgroup_size : int, default=10
        Maximum number of attributes to consider in subgroup definitions
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'final_classifier': trained fair classifier
        - 'game_value': final value of the zero-sum game
        - 'convergence_history': list of game values over iterations
        - 'worst_subgroup_violation': maximum fairness violation found
        - 'subgroup_violations': violations for each discovered subgroup
        - 'learner_regret': cumulative regret of learner
        - 'auditor_regret': cumulative regret of auditor
        - 'n_iterations': number of iterations until convergence
        
    Raises
    ------
    ValueError
        If inputs have incompatible shapes or invalid constraint type
    """
    
    # Input validation
    X = np.asarray(X)
    y = np.asarray(y)
    protected_attributes = np.asarray(protected_attributes)
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    
    if X.shape[0] != protected_attributes.shape[0]:
        raise ValueError("X and protected_attributes must have same number of samples")
    
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must be binary (0 or 1)")
    
    if fairness_constraint not in ['fpr', 'fnr']:
        raise ValueError("fairness_constraint must be 'fpr' or 'fnr'")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples, n_features = X.shape
    n_protected = protected_attributes.shape[1] if protected_attributes.ndim > 1 else 1
    
    if protected_attributes.ndim == 1:
        protected_attributes = protected_attributes.reshape(-1, 1)
    
    # Initialize base classifier
    if base_classifier is None:
        base_classifier = LogisticRegression(random_state=random_state)
    
    # Generate all possible subgroups using linear threshold functions
    # Each subgroup is defined by linear inequalities over protected attributes
    def generate_subgroups(protected_attrs: np.ndarray, max_size: int) -> List[Dict]:
        """Generate subgroups using linear threshold functions over protected attributes"""
        subgroups = []
        n_attrs = protected_attrs.shape[1]
        
        # Generate random linear threshold functions
        for _ in range(min(max_size, 2**n_attrs)):
            # Random coefficients for linear combination
            coeffs = np.random.randn(n_attrs)
            # Random threshold
            threshold = np.random.randn()
            
            # Define subgroup membership
            linear_combo = protected_attrs @ coeffs
            membership = linear_combo >= threshold
            
            if np.sum(membership) > 0 and np.sum(membership) < len(membership):
                subgroups.append({
                    'coefficients': coeffs,
                    'threshold': threshold,
                    'membership': membership,
                    'size': np.sum(membership)
                })
        
        return subgroups
    
    def compute_fairness_violation(y_true: np.ndarray, y_pred: np.ndarray, 
                                 membership: np.ndarray, constraint: str) -> float:
        """Compute fairness violation for a subgroup"""
        if np.sum(membership) == 0:
            return 0.0
        
        # Overall rates
        if constraint == 'fpr':
            # False Positive Rate
            overall_negatives = np.sum(y_true == 0)
            overall_fpr = np.sum((y_pred == 1) & (y_true == 0)) / max(overall_negatives, 1)
            
            # Subgroup FPR
            subgroup_negatives = np.sum((y_true == 0) & membership)
            subgroup_fpr = np.sum((y_pred == 1) & (y_true == 0) & membership) / max(subgroup_negatives, 1)
            
            return abs(subgroup_fpr - overall_fpr)
        
        elif constraint == 'fnr':
            # False Negative Rate
            overall_positives = np.sum(y_true == 1)
            overall_fnr = np.sum((y_pred == 0) & (y_true == 1)) / max(overall_positives, 1)
            
            # Subgroup FNR
            subgroup_positives = np.sum((y_true == 1) & membership)
            subgroup_fnr = np.sum((y_pred == 0) & (y_true == 1) & membership) / max(subgroup_positives, 1)
            
            return abs(subgroup_fnr - overall_fnr)
    
    # Initialize Fictitious Play algorithm
    # Learner maintains distribution over classifiers
    # Auditor maintains distribution over subgroups
    
    classifiers_history = []
    subgroups_history = []
    learner_weights = []
    auditor_weights = []
    game_values = []
    
    # Generate initial subgroups
    current_subgroups = generate_subgroups(protected_attributes, max_subgroup_size)
    
    for iteration in range(max_iterations):
        # Learner's turn: train classifier against current auditor strategy
        if iteration == 0:
            # Initial uniform weights over subgroups
            auditor_dist = np.ones(len(current_subgroups)) / len(current_subgroups)
        else:
            # Update auditor distribution based on fictitious play
            auditor_weights_array = np.array(auditor_weights)
            auditor_dist = np.mean(auditor_weights_array, axis=0)
        
        # Train classifier with weighted fairness constraints
        classifier = base_classifier.__class__(**base_classifier.get_params())
        
        # Simple approach: train on reweighted samples based on subgroup violations
        sample_weights = np.ones(n_samples)
        for i, subgroup in enumerate(current_subgroups):
            if auditor_dist[i] > 0:
                sample_weights[subgroup['membership']] += auditor_dist[i]
        
        sample_weights /= np.sum(sample_weights) / n_samples
        
        try:
            classifier.fit(X, y, sample_weight=sample_weights)
        except TypeError:
            # Classifier doesn't support sample weights
            classifier.fit(X, y)
        
        classifiers_history.append(classifier)
        y_pred = classifier.predict(X)
        
        # Auditor's turn: find worst subgroup violation
        violations = []
        for subgroup in current_subgroups:
            violation = compute_fairness_violation(y, y_pred, subgroup['membership'], fairness_constraint)
            violations.append(violation)
        
        violations = np.array(violations)
        
        # Auditor chooses subgroup with maximum violation
        best_subgroup_idx = np.argmax(violations)
        auditor_strategy = np.zeros(len(current_subgroups))
        auditor_strategy[best_subgroup_idx] = 1.0
        
        auditor_weights.append(auditor_strategy)
        
        # Learner updates strategy (uniform over all past classifiers)
        learner_strategy = np.ones(len(classifiers_history)) / len(classifiers_history)
        learner_weights.append(learner_strategy)
        
        # Compute game value (maximum violation under current strategies)
        game_value = np.max(violations)
        game_values.append(game_value)
        
        # Check convergence
        if iteration > 0 and abs(game_values[-1] - game_values[-2]) < tolerance:
            break
        
        # Periodically generate new subgroups to avoid local optima
        if iteration % 20 == 19:
            new_subgroups = generate_subgroups(protected_attributes, max_subgroup_size // 2)
            current_subgroups.extend(new_subgroups)
            # Extend auditor weights with zeros for new subgroups
            for j in range(len(auditor_weights)):
                auditor_weights[j] = np.concatenate([auditor_weights[j], 
                                                   np.zeros(len(new_subgroups))])
    
    # Create final fair classifier as mixture of all classifiers
    final_learner_dist = np.array(learner_weights[-1]) if learner_weights else np.array([1.0])
    
    # For simplicity, return the last classifier (could implement proper mixture)
    final_classifier = classifiers_history[-1] if classifiers_history else base_classifier
    
    # Compute final statistics
    final_predictions = final_classifier.predict(X)
    
    subgroup_violations = []
    for i, subgroup in enumerate(current_subgroups):
        violation = compute_fairness_violation(y, final_predictions, subgroup['membership'], fairness_constraint)
        subgroup_violations.append({
            'subgroup_id': i,
            'size': subgroup['size'],
            'violation': violation,
            'coefficients': subgroup['coefficients'],
            'threshold': subgroup['threshold']
        })
    
    worst_violation = max([sv['violation'] for sv in subgroup_violations]) if subgroup_violations else 0.0
    
    # Compute regrets (simplifie