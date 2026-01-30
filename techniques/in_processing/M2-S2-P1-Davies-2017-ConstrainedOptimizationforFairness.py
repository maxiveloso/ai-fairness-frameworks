import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize
import warnings

def constrained_optimization_for_fairness(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    sensitive_attribute: Union[np.ndarray, pd.Series],
    base_estimator: Optional[BaseEstimator] = None,
    fairness_constraint: str = "equal_opportunity",
    constraint_tolerance: float = 0.01,
    max_iterations: int = 1000,
    learning_rate: float = 0.01,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Implement constrained optimization for fairness using Lagrangian dual ascent method.
    
    This technique addresses algorithmic fairness by optimizing model predictions subject to
    fairness constraints. It uses constrained optimization to ensure equal treatment across
    different demographic groups while maintaining predictive performance.
    
    Statistical Concepts:
    - Equal Opportunity: TPR should be equal across groups (TPR_a = TPR_b)
    - Equalized Odds: Both TPR and FPR should be equal across groups
    - Lagrangian Optimization: Uses dual variables to enforce constraints
    - Group-specific Thresholds: Different decision thresholds for different groups
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Binary target variable (0 or 1)
    sensitive_attribute : array-like of shape (n_samples,)
        Binary sensitive attribute (e.g., gender, race)
    base_estimator : BaseEstimator, optional
        Base classifier to use. Default is LogisticRegression
    fairness_constraint : str, default="equal_opportunity"
        Type of fairness constraint ("equal_opportunity" or "equalized_odds")
    constraint_tolerance : float, default=0.01
        Tolerance for constraint violations
    max_iterations : int, default=1000
        Maximum number of optimization iterations
    learning_rate : float, default=0.01
        Learning rate for dual ascent algorithm
    random_state : int, optional
        Random state for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'optimized_thresholds': Group-specific decision thresholds
        - 'fairness_metrics': TPR, FPR, and other metrics by group
        - 'constraint_violations': Degree of constraint violation
        - 'convergence_info': Information about optimization convergence
        - 'predictions': Final fair predictions
        - 'base_predictions': Original model predictions before fairness adjustment
    """
    
    # Input validation
    X = np.asarray(X)
    y = np.asarray(y)
    sensitive_attribute = np.asarray(sensitive_attribute)
    
    if X.shape[0] != len(y) or X.shape[0] != len(sensitive_attribute):
        raise ValueError("X, y, and sensitive_attribute must have the same number of samples")
    
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must be binary (0 or 1)")
    
    if not np.all(np.isin(sensitive_attribute, [0, 1])):
        raise ValueError("sensitive_attribute must be binary (0 or 1)")
    
    if fairness_constraint not in ["equal_opportunity", "equalized_odds"]:
        raise ValueError("fairness_constraint must be 'equal_opportunity' or 'equalized_odds'")
    
    # Initialize base estimator
    if base_estimator is None:
        base_estimator = LogisticRegression(random_state=random_state)
    
    # Train base model to get probability predictions
    base_estimator.fit(X, y)
    prob_predictions = base_estimator.predict_proba(X)[:, 1]
    base_predictions = base_estimator.predict(X)
    
    # Separate groups based on sensitive attribute
    group_0_mask = sensitive_attribute == 0
    group_1_mask = sensitive_attribute == 1
    
    def calculate_rates(predictions, true_labels, group_mask):
        """Calculate TPR and FPR for a specific group"""
        if np.sum(group_mask) == 0:
            return 0.0, 0.0
        
        group_pred = predictions[group_mask]
        group_true = true_labels[group_mask]
        
        if len(np.unique(group_true)) < 2:
            # Handle case where group has only one class
            if np.all(group_true == 1):
                tpr = np.mean(group_pred == 1)
                fpr = 0.0
            else:
                tpr = 0.0
                fpr = np.mean(group_pred == 1)
        else:
            tn, fp, fn, tp = confusion_matrix(group_true, group_pred, labels=[0, 1]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return tpr, fpr
    
    def objective_function(thresholds):
        """Objective function: maximize utility while satisfying fairness constraints"""
        threshold_0, threshold_1 = thresholds
        
        # Generate predictions using group-specific thresholds
        predictions = np.zeros_like(y)
        predictions[group_0_mask] = (prob_predictions[group_0_mask] >= threshold_0).astype(int)
        predictions[group_1_mask] = (prob_predictions[group_1_mask] >= threshold_1).astype(int)
        
        # Calculate accuracy as utility measure
        accuracy = np.mean(predictions == y)
        
        # Calculate fairness constraint violations
        tpr_0, fpr_0 = calculate_rates(predictions, y, group_0_mask)
        tpr_1, fpr_1 = calculate_rates(predictions, y, group_1_mask)
        
        # Constraint violations (we want to minimize these)
        tpr_violation = abs(tpr_0 - tpr_1)
        
        if fairness_constraint == "equalized_odds":
            fpr_violation = abs(fpr_0 - fpr_1)
            total_violation = tpr_violation + fpr_violation
        else:
            total_violation = tpr_violation
        
        # Return negative accuracy plus penalty for constraint violations
        # Higher penalty weight ensures constraints are prioritized
        penalty_weight = 10.0
        return -accuracy + penalty_weight * total_violation
    
    # Optimize thresholds using constrained optimization
    initial_thresholds = [0.5, 0.5]  # Start with equal thresholds
    bounds = [(0.0, 1.0), (0.0, 1.0)]  # Thresholds must be between 0 and 1
    
    try:
        result = minimize(
            objective_function,
            initial_thresholds,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations}
        )
        
        optimal_thresholds = result.x
        converged = result.success
        
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}. Using equal thresholds.")
        optimal_thresholds = [0.5, 0.5]
        converged = False
    
    # Generate final predictions using optimal thresholds
    final_predictions = np.zeros_like(y)
    final_predictions[group_0_mask] = (prob_predictions[group_0_mask] >= optimal_thresholds[0]).astype(int)
    final_predictions[group_1_mask] = (prob_predictions[group_1_mask] >= optimal_thresholds[1]).astype(int)
    
    # Calculate final fairness metrics
    tpr_0_final, fpr_0_final = calculate_rates(final_predictions, y, group_0_mask)
    tpr_1_final, fpr_1_final = calculate_rates(final_predictions, y, group_1_mask)
    
    # Calculate constraint violations
    tpr_violation_final = abs(tpr_0_final - tpr_1_final)
    fpr_violation_final = abs(fpr_0_final - fpr_1_final)
    
    # Calculate baseline metrics (before fairness adjustment)
    tpr_0_base, fpr_0_base = calculate_rates(base_predictions, y, group_0_mask)
    tpr_1_base, fpr_1_base = calculate_rates(base_predictions, y, group_1_mask)
    
    return {
        'optimized_thresholds': {
            'group_0': optimal_thresholds[0],
            'group_1': optimal_thresholds[1]
        },
        'fairness_metrics': {
            'group_0': {
                'tpr': tpr_0_final,
                'fpr': fpr_0_final,
                'sample_size': np.sum(group_0_mask)
            },
            'group_1': {
                'tpr': tpr_1_final,
                'fpr': fpr_1_final,
                'sample_size': np.sum(group_1_mask)
            }
        },
        'constraint_violations': {
            'tpr_difference': tpr_violation_final,
            'fpr_difference': fpr_violation_final,
            'satisfies_equal_opportunity': tpr_violation_final <= constraint_tolerance,
            'satisfies_equalized_odds': (tpr_violation_final <= constraint_tolerance and 
                                       fpr_violation_final <= constraint_tolerance)
        },
        'convergence_info': {
            'converged': converged,
            'constraint_type': fairness_constraint,
            'tolerance': constraint_tolerance
        },
        'predictions': final_predictions,
        'base_predictions': base_predictions,
        'baseline_metrics': {
            'group_0': {'tpr': tpr_0_base, 'fpr': fpr_0_base},
            'group_1': {'tpr': tpr_1_base, 'fpr': fpr_1_base},
            'tpr_difference': abs(tpr_0_base - tpr_1_base),
            'fpr_difference': abs(fpr_0_base - fpr_1_base)
        }
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic dataset with bias
    n_samples = 1000
    n_features = 5
    
    # Create features
    X = np.random.randn(n_samples, n_features)
    
    # Create sensitive attribute (e.g., gender: 0=female, 1=male)
    sensitive_attr = np.random.binomial(1, 0.4, n_samples)
    
    # Create biased target variable
    # The model will have different base rates for different groups
    linear_combination = X.dot(np.random.randn(n_features))
    bias_term = sensitive_attr * 0.5  # Bias favoring group 1
    y_prob = 1 / (1 + np.exp(-(linear_combination + bias_term)))
    y = np.random.binomial(1, y_prob)
    
    print("Constrained Optimization for Fairness - Example")
    print("=" * 50)
    
    # Test