import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Union, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
import warnings

def multiaccuracy_boost(
    predictor: Callable[[np.ndarray], np.ndarray],
    X_audit: np.ndarray,
    y_audit: np.ndarray,
    subgroup_functions: List[Callable[[np.ndarray], np.ndarray]],
    alpha: float = 0.01,
    beta: float = 0.01,
    max_iterations: int = 100,
    hypothesis_class: Optional[BaseEstimator] = None,
    convergence_threshold: float = 1e-6,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Implements the MULTIACCURACY-BOOST algorithm for post-processing predictions
    to achieve multiaccuracy across specified subgroups.
    
    Multiaccuracy ensures that a predictor is well-calibrated not just on the overall
    population, but also on every subgroup defined by the hypothesis class. The algorithm
    iteratively identifies miscalibrated subgroups and applies corrections to improve
    calibration across all groups simultaneously.
    
    Parameters
    ----------
    predictor : Callable[[np.ndarray], np.ndarray]
        Black-box predictor function that takes features and returns probability predictions
    X_audit : np.ndarray
        Audit dataset features of shape (n_samples, n_features)
    y_audit : np.ndarray
        True binary labels for audit dataset of shape (n_samples,)
    subgroup_functions : List[Callable[[np.ndarray], np.ndarray]]
        List of functions that define subgroups, each taking X and returning boolean mask
    alpha : float, default=0.01
        Accuracy parameter - maximum allowed calibration error within subgroups
    beta : float, default=0.01
        Fairness parameter - minimum subgroup size as fraction of total population
    max_iterations : int, default=100
        Maximum number of boosting iterations
    hypothesis_class : BaseEstimator, optional
        Sklearn-compatible weak learner for identifying miscalibrated subgroups.
        If None, uses DecisionTreeClassifier with max_depth=3
    convergence_threshold : float, default=1e-6
        Threshold for convergence based on loss improvement
    verbose : bool, default=False
        Whether to print iteration details
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'corrected_predictions': Final multiaccurate predictions
        - 'original_predictions': Original predictor outputs
        - 'iterations': Number of iterations performed
        - 'converged': Whether algorithm converged
        - 'loss_history': Cross-entropy loss at each iteration
        - 'subgroup_calibration_errors': Final calibration errors per subgroup
        - 'overall_calibration_error': Overall calibration error
        - 'correction_weights': Learned correction weights from boosting
        
    Raises
    ------
    ValueError
        If inputs have incompatible shapes or invalid parameter values
    TypeError
        If predictor is not callable or subgroup_functions is not a list
        
    Notes
    -----
    The algorithm works by:
    1. Starting with original predictor outputs
    2. At each iteration, finding the most miscalibrated subgroup using hypothesis class
    3. Learning a correction for that subgroup
    4. Updating predictions with weighted correction
    5. Repeating until all subgroups are well-calibrated or max iterations reached
    
    The theoretical guarantee is that after T iterations, all subgroups will have
    calibration error at most alpha + O(sqrt(log(|H|)/n)) where |H| is the size
    of the hypothesis class and n is the audit dataset size.
    
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> 
    >>> # Create sample data
    >>> np.random.seed(42)
    >>> X = np.random.randn(1000, 5)
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> 
    >>> # Train a potentially biased predictor
    >>> model = LogisticRegression()
    >>> model.fit(X, y)
    >>> predictor = lambda x: model.predict_proba(x)[:, 1]
    >>> 
    >>> # Define subgroups (e.g., based on sensitive attributes)
    >>> def subgroup_1(X): return X[:, 2] > 0  # Feature 2 positive
    >>> def subgroup_2(X): return X[:, 3] < -0.5  # Feature 3 very negative
    >>> subgroups = [subgroup_1, subgroup_2]
    >>> 
    >>> # Apply multiaccuracy boosting
    >>> results = multiaccuracy_boost(predictor, X, y, subgroups, alpha=0.05)
    >>> print(f"Converged: {results['converged']}")
    >>> print(f"Iterations: {results['iterations']}")
    """
    
    # Input validation
    if not callable(predictor):
        raise TypeError("predictor must be callable")
    
    if not isinstance(subgroup_functions, list):
        raise TypeError("subgroup_functions must be a list")
    
    X_audit = np.asarray(X_audit)
    y_audit = np.asarray(y_audit)
    
    if X_audit.ndim != 2:
        raise ValueError("X_audit must be 2-dimensional")
    
    if y_audit.ndim != 1:
        raise ValueError("y_audit must be 1-dimensional")
    
    if len(X_audit) != len(y_audit):
        raise ValueError("X_audit and y_audit must have same number of samples")
    
    if not np.all(np.isin(y_audit, [0, 1])):
        raise ValueError("y_audit must contain only binary labels (0, 1)")
    
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    
    if not 0 < beta < 1:
        raise ValueError("beta must be between 0 and 1")
    
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    
    n_samples = len(X_audit)
    min_subgroup_size = int(beta * n_samples)
    
    # Initialize hypothesis class
    if hypothesis_class is None:
        hypothesis_class = DecisionTreeClassifier(max_depth=3, random_state=42)
    
    # Get original predictions
    try:
        original_predictions = predictor(X_audit)
    except Exception as e:
        raise ValueError(f"Error calling predictor: {e}")
    
    original_predictions = np.asarray(original_predictions)
    if original_predictions.ndim != 1 or len(original_predictions) != n_samples:
        raise ValueError("predictor must return 1D array with same length as X_audit")
    
    # Clip predictions to avoid log(0) issues
    original_predictions = np.clip(original_predictions, 1e-10, 1 - 1e-10)
    
    # Initialize boosting
    current_predictions = original_predictions.copy()
    correction_weights = []
    loss_history = []
    
    # Calculate initial loss
    initial_loss = log_loss(y_audit, current_predictions)
    loss_history.append(initial_loss)
    
    if verbose:
        print(f"Initial loss: {initial_loss:.6f}")
    
    converged = False
    iteration = 0
    
    for iteration in range(max_iterations):
        # Find most miscalibrated subgroup
        max_violation = 0
        best_subgroup_mask = None
        best_subgroup_idx = -1
        
        # Check predefined subgroups
        for i, subgroup_func in enumerate(subgroup_functions):
            try:
                mask = subgroup_func(X_audit)
                mask = np.asarray(mask, dtype=bool)
            except Exception as e:
                warnings.warn(f"Error in subgroup function {i}: {e}")
                continue
            
            if np.sum(mask) < min_subgroup_size:
                continue
            
            # Calculate calibration error for this subgroup
            subgroup_predictions = current_predictions[mask]
            subgroup_labels = y_audit[mask]
            
            if len(subgroup_predictions) == 0:
                continue
            
            # Calibration error = |E[Y|subgroup] - E[predictions|subgroup]|
            calibration_error = abs(np.mean(subgroup_labels) - np.mean(subgroup_predictions))
            
            if calibration_error > max_violation:
                max_violation = calibration_error
                best_subgroup_mask = mask
                best_subgroup_idx = i
        
        # Use hypothesis class to find additional miscalibrated subgroups
        residuals = y_audit - current_predictions
        
        try:
            # Fit weak learner to predict residuals
            weak_learner = hypothesis_class.__class__(**hypothesis_class.get_params())
            weak_learner.fit(X_audit, residuals)
            
            # Get predictions from weak learner
            weak_predictions = weak_learner.predict(X_audit)
            
            # Find subgroup where weak learner predicts large residuals
            threshold = np.percentile(np.abs(weak_predictions), 75)
            hypothesis_mask = np.abs(weak_predictions) >= threshold
            
            if np.sum(hypothesis_mask) >= min_subgroup_size:
                subgroup_predictions = current_predictions[hypothesis_mask]
                subgroup_labels = y_audit[hypothesis_mask]
                
                calibration_error = abs(np.mean(subgroup_labels) - np.mean(subgroup_predictions))
                
                if calibration_error > max_violation:
                    max_violation = calibration_error
                    best_subgroup_mask = hypothesis_mask
                    best_subgroup_idx = -1  # Indicates hypothesis class found this
                    
        except Exception as e:
            if verbose:
                print(f"Warning: Hypothesis class failed at iteration {iteration}: {e}")
        
        # Check convergence
        if max_violation <= alpha:
            converged = True
            if verbose:
                print(f"Converged at iteration {iteration}: max violation {max_violation:.6f} <= {alpha}")
            break
        
        if best_subgroup_mask is None:
            if verbose:
                print(f"No suitable subgroup found at iteration {iteration}")
            break
        
        # Calculate correction for best subgroup
        subgroup_predictions = current_predictions[best_subgroup_mask]
        subgroup_labels = y_audit[best_subgroup_mask]
        
        # Correction is the difference between actual and predicted rates
        correction = np.mean(subgroup_labels) - np.mean(subgroup_predictions)
        
        # Apply correction with learning rate (using 1/sqrt(t+1) schedule)
        learning_rate = 1.0 / np.sqrt(iteration + 2)
        correction_weight = learning_rate * correction
        
        # Update predictions
        current_predictions[best_subgroup_mask] += correction_weight
        current_predictions = np.clip(current_predictions, 1e-10, 1 - 1e-10)
        
        # Store correction information
        correction_weights.append({
            'iteration': iteration,
            'subgroup_idx': best_subgroup_idx,
            'mask': best_subgroup_mask.copy(),
            'correction': correction,
            'weight': correction_weight,
            'violation': max_violation
        })
        
        # Calculate new loss
        new_loss = log_loss(y_audit, current_predictions)
        loss_