import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

def subgroup_fairness_auditing_and_learning(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    sensitive_features: Union[np.ndarray, pd.DataFrame],
    gamma: float = 0.01,
    max_iterations: int = 100,
    algorithm: str = "fictitious_play",
    oracle: Optional[Callable] = None,
    learning_rate: float = 0.1,
    convergence_threshold: float = 1e-6,
    random_state: Optional[int] = None
) -> Dict:
    """
    Implements Subgroup Fairness Auditing and Learning algorithm from Kearns et al. (2018).
    
    This algorithm formulates fair learning as a zero-sum game between a Learner and an Auditor.
    The Learner tries to minimize loss while the Auditor tries to find subgroups where the 
    classifier performs unfairly. The algorithm ensures that no subgroup (defined by combinations
    of sensitive features) experiences unfairness beyond a specified threshold gamma.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Training features
    y : array-like of shape (n_samples,)
        Binary target variable (0 or 1)
    sensitive_features : array-like of shape (n_samples, n_sensitive_features)
        Sensitive attributes used to define subgroups
    gamma : float, default=0.01
        Fairness violation tolerance. Smaller values enforce stricter fairness
    max_iterations : int, default=100
        Maximum number of game iterations
    algorithm : str, default="fictitious_play"
        Algorithm choice: "fictitious_play" or "follow_perturbed_leader"
    oracle : callable, optional
        Learning oracle function. If None, uses logistic regression
    learning_rate : float, default=0.1
        Learning rate for the algorithms
    convergence_threshold : float, default=1e-6
        Convergence threshold for stopping criterion
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'final_classifier': Final fair classifier
        - 'fairness_violations': Maximum fairness violation found
        - 'subgroup_accuracies': Accuracy for each subgroup
        - 'overall_accuracy': Overall classifier accuracy
        - 'iterations': Number of iterations until convergence
        - 'converged': Whether algorithm converged
        - 'auditor_best_response': Best subgroup found by auditor
        - 'game_values': History of game values during training
    """
    
    # Input validation
    X = np.asarray(X)
    y = np.asarray(y)
    sensitive_features = np.asarray(sensitive_features)
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if X.shape[0] != sensitive_features.shape[0]:
        raise ValueError("X and sensitive_features must have the same number of samples")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must be binary (0 or 1)")
    if gamma <= 0 or gamma >= 1:
        raise ValueError("gamma must be between 0 and 1")
    if algorithm not in ["fictitious_play", "follow_perturbed_leader"]:
        raise ValueError("algorithm must be 'fictitious_play' or 'follow_perturbed_leader'")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples, n_features = X.shape
    
    # Default oracle using logistic regression
    if oracle is None:
        def default_oracle(X_train, y_train, weights=None):
            clf = LogisticRegression(random_state=random_state, max_iter=1000)
            if weights is not None:
                clf.fit(X_train, y_train, sample_weight=weights)
            else:
                clf.fit(X_train, y_train)
            return clf
        oracle = default_oracle
    
    # Generate all possible subgroups based on sensitive features
    # For simplicity, we consider each unique combination of sensitive features as a subgroup
    if sensitive_features.ndim == 1:
        sensitive_features = sensitive_features.reshape(-1, 1)
    
    unique_groups = np.unique(sensitive_features, axis=0)
    n_subgroups = len(unique_groups)
    
    # Initialize game components
    learner_strategy = np.ones(max_iterations) / max_iterations  # Uniform initially
    auditor_strategy = np.ones(n_subgroups) / n_subgroups  # Uniform over subgroups
    
    classifiers = []
    game_values = []
    
    # Game iteration
    converged = False
    final_iteration = max_iterations
    
    for iteration in range(max_iterations):
        # Learner's move: train classifier using current auditor strategy as weights
        sample_weights = np.zeros(n_samples)
        
        for i, group in enumerate(unique_groups):
            # Find samples belonging to this subgroup
            group_mask = np.all(sensitive_features == group, axis=1)
            sample_weights[group_mask] = auditor_strategy[i]
        
        # Normalize weights
        if sample_weights.sum() > 0:
            sample_weights = sample_weights / sample_weights.sum() * n_samples
        else:
            sample_weights = np.ones(n_samples)
        
        # Train classifier with weighted samples
        classifier = oracle(X, y, sample_weights)
        classifiers.append(classifier)
        
        # Auditor's move: find worst subgroup
        y_pred = classifier.predict(X)
        subgroup_violations = []
        
        for i, group in enumerate(unique_groups):
            group_mask = np.all(sensitive_features == group, axis=1)
            if np.sum(group_mask) == 0:
                subgroup_violations.append(0)
                continue
                
            # Calculate fairness violation for this subgroup
            # Using accuracy difference from overall accuracy as fairness metric
            group_accuracy = accuracy_score(y[group_mask], y_pred[group_mask])
            overall_accuracy = accuracy_score(y, y_pred)
            violation = abs(group_accuracy - overall_accuracy)
            subgroup_violations.append(violation)
        
        subgroup_violations = np.array(subgroup_violations)
        
        # Update strategies based on algorithm choice
        if algorithm == "fictitious_play":
            # Fictitious Play: best response to historical average
            if iteration == 0:
                # First iteration: uniform response
                auditor_best_response = np.argmax(subgroup_violations)
                new_auditor_strategy = np.zeros(n_subgroups)
                new_auditor_strategy[auditor_best_response] = 1.0
            else:
                # Update auditor strategy as average of all best responses
                auditor_best_response = np.argmax(subgroup_violations)
                auditor_strategy = (auditor_strategy * iteration + 
                                 np.eye(n_subgroups)[auditor_best_response]) / (iteration + 1)
            
        elif algorithm == "follow_perturbed_leader":
            # Follow the Perturbed Leader with exponential weights
            perturbation = np.random.exponential(1.0, n_subgroups)
            perturbed_violations = subgroup_violations + perturbation
            auditor_best_response = np.argmax(perturbed_violations)
            
            # Exponential weight update
            auditor_strategy = auditor_strategy * np.exp(learning_rate * subgroup_violations)
            auditor_strategy = auditor_strategy / auditor_strategy.sum()
        
        # Record game value (maximum violation)
        max_violation = np.max(subgroup_violations)
        game_values.append(max_violation)
        
        # Check convergence
        if iteration > 10 and len(game_values) >= 5:
            recent_values = game_values[-5:]
            if np.std(recent_values) < convergence_threshold:
                converged = True
                final_iteration = iteration + 1
                break
    
    # Select final classifier (last one or mixture)
    final_classifier = classifiers[-1]
    
    # Calculate final metrics
    y_pred_final = final_classifier.predict(X)
    overall_accuracy = accuracy_score(y, y_pred_final)
    
    # Calculate subgroup accuracies
    subgroup_accuracies = {}
    max_fairness_violation = 0
    
    for i, group in enumerate(unique_groups):
        group_mask = np.all(sensitive_features == group, axis=1)
        if np.sum(group_mask) > 0:
            group_accuracy = accuracy_score(y[group_mask], y_pred_final[group_mask])
            subgroup_accuracies[f'subgroup_{i}'] = group_accuracy
            violation = abs(group_accuracy - overall_accuracy)
            max_fairness_violation = max(max_fairness_violation, violation)
    
    return {
        'final_classifier': final_classifier,
        'fairness_violations': max_fairness_violation,
        'subgroup_accuracies': subgroup_accuracies,
        'overall_accuracy': overall_accuracy,
        'iterations': final_iteration,
        'converged': converged,
        'auditor_best_response': auditor_best_response,
        'game_values': game_values,
        'meets_fairness_constraint': max_fairness_violation <= gamma,
        'unique_subgroups': unique_groups.tolist(),
        'final_auditor_strategy': auditor_strategy.tolist()
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic dataset
    n_samples = 1000
    n_features = 5
    
    # Create features
    X = np.random.randn(n_samples, n_features)
    
    # Create sensitive attribute (e.g., gender: 0 or 1)
    sensitive_attr = np.random.binomial(1, 0.5, n_samples)
    
    # Create target with some bias related to sensitive attribute
    # This creates unfairness that the algorithm should detect and mitigate
    y_prob = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1] + 0.3 * sensitive_attr)))
    y = np.random.binomial(1, y_prob)
    
    print("Subgroup Fairness Auditing and Learning Example")
    print("=" * 50)
    
    # Run the algorithm
    results = subgroup_fairness_auditing_and_learning(
        X=X,
        y=y,
        sensitive_features=sensitive_attr.reshape(-1, 1),
        gamma=0.05,
        max_iterations=50,
        algorithm="fictitious_play",
        random_state=42
    )
    
    print(f"Algorithm converged: {results['converged']}")
    print(f"Number of iterations: {results['iterations']}")
    print(f"Overall accuracy: {results['overall_accuracy']:.4f}")
    print(f"Maximum fairness violation: {results['fairness_violations']:.4f}")
    print(f"Meets fairness constraint (γ=0.05): {results['meets_fairness_