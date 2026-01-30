import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, Callable
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from scipy.optimize import minimize_scalar
import warnings

def fairness_accuracy_tradeoff_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive_attr: np.ndarray,
    fairness_metric: str = 'demographic_parity',
    cost_ratio: float = 1.0,
    threshold_method: str = 'bayes_optimal',
    return_thresholds: bool = True
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Analyze the fairness-accuracy trade-off using cost-sensitive risk formulation.
    
    This implementation follows Menon & Williamson (2018) approach to quantify
    the trade-off between fairness and accuracy in binary classification through
    cost-sensitive learning and instance-dependent thresholding.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_prob : np.ndarray
        Predicted probabilities for positive class
    sensitive_attr : np.ndarray
        Binary sensitive attribute (0 or 1, e.g., gender, race)
    fairness_metric : str, default='demographic_parity'
        Fairness constraint type ('demographic_parity' or 'equalized_odds')
    cost_ratio : float, default=1.0
        Ratio of fairness violation cost to accuracy loss cost
    threshold_method : str, default='bayes_optimal'
        Method for threshold selection ('bayes_optimal' or 'single_threshold')
    return_thresholds : bool, default=True
        Whether to return optimal thresholds for each group
        
    Returns
    -------
    Dict[str, Union[float, np.ndarray, Dict]]
        Dictionary containing:
        - 'fairness_violation': Fairness constraint violation measure
        - 'accuracy_unconstrained': Accuracy without fairness constraints
        - 'accuracy_constrained': Accuracy with fairness constraints
        - 'cost_sensitive_risk': Total cost-sensitive risk
        - 'bayes_optimal_risk': Theoretical minimum risk
        - 'fairness_accuracy_ratio': Ratio of fairness violation to accuracy loss
        - 'group_statistics': Per-group performance statistics
        - 'optimal_thresholds': Optimal decision thresholds (if return_thresholds=True)
        
    Notes
    -----
    The cost-sensitive risk formulation balances accuracy and fairness:
    R(h) = R_acc(h) + λ * R_fair(h)
    
    where R_acc is accuracy risk, R_fair is fairness violation risk,
    and λ (cost_ratio) controls the trade-off.
    
    For demographic parity: P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
    For equalized odds: P(Ŷ=1|Y=y,A=0) = P(Ŷ=1|Y=y,A=1) for y ∈ {0,1}
    """
    
    # Input validation
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    sensitive_attr = np.asarray(sensitive_attr)
    
    if len(y_true) != len(y_prob) or len(y_true) != len(sensitive_attr):
        raise ValueError("All input arrays must have the same length")
    
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1")
        
    if not np.all((y_prob >= 0) & (y_prob <= 1)):
        raise ValueError("y_prob must be between 0 and 1")
        
    if not np.all(np.isin(sensitive_attr, [0, 1])):
        raise ValueError("sensitive_attr must contain only 0 and 1")
        
    if fairness_metric not in ['demographic_parity', 'equalized_odds']:
        raise ValueError("fairness_metric must be 'demographic_parity' or 'equalized_odds'")
    
    # Group indices
    group_0_idx = sensitive_attr == 0
    group_1_idx = sensitive_attr == 1
    
    if not (np.any(group_0_idx) and np.any(group_1_idx)):
        raise ValueError("Both groups must be present in sensitive_attr")
    
    # Compute unconstrained accuracy (using single threshold = 0.5)
    y_pred_unconstrained = (y_prob >= 0.5).astype(int)
    accuracy_unconstrained = np.mean(y_true == y_pred_unconstrained)
    
    def _compute_fairness_violation(y_pred: np.ndarray) -> float:
        """Compute fairness constraint violation."""
        if fairness_metric == 'demographic_parity':
            # |P(Ŷ=1|A=0) - P(Ŷ=1|A=1)|
            rate_0 = np.mean(y_pred[group_0_idx])
            rate_1 = np.mean(y_pred[group_1_idx])
            return abs(rate_0 - rate_1)
        
        elif fairness_metric == 'equalized_odds':
            # |P(Ŷ=1|Y=1,A=0) - P(Ŷ=1|Y=1,A=1)| + |P(Ŷ=1|Y=0,A=0) - P(Ŷ=1|Y=0,A=1)|
            violation = 0.0
            
            for y_val in [0, 1]:
                mask_0 = group_0_idx & (y_true == y_val)
                mask_1 = group_1_idx & (y_true == y_val)
                
                if np.any(mask_0) and np.any(mask_1):
                    rate_0 = np.mean(y_pred[mask_0])
                    rate_1 = np.mean(y_pred[mask_1])
                    violation += abs(rate_0 - rate_1)
            
            return violation
    
    def _cost_sensitive_risk(thresholds: Union[float, Tuple[float, float]]) -> float:
        """Compute cost-sensitive risk for given threshold(s)."""
        if isinstance(thresholds, (int, float)):
            # Single threshold for both groups
            y_pred = (y_prob >= thresholds).astype(int)
        else:
            # Group-specific thresholds
            thresh_0, thresh_1 = thresholds
            y_pred = np.zeros_like(y_true)
            y_pred[group_0_idx] = (y_prob[group_0_idx] >= thresh_0).astype(int)
            y_pred[group_1_idx] = (y_prob[group_1_idx] >= thresh_1).astype(int)
        
        # Accuracy risk (1 - accuracy)
        accuracy_risk = 1.0 - np.mean(y_true == y_pred)
        
        # Fairness violation risk
        fairness_risk = _compute_fairness_violation(y_pred)
        
        # Total cost-sensitive risk
        return accuracy_risk + cost_ratio * fairness_risk
    
    # Optimize thresholds based on method
    if threshold_method == 'single_threshold':
        # Single threshold optimization
        result = minimize_scalar(_cost_sensitive_risk, bounds=(0, 1), method='bounded')
        optimal_threshold = result.x
        optimal_thresholds = {'group_0': optimal_threshold, 'group_1': optimal_threshold}
        
        y_pred_constrained = (y_prob >= optimal_threshold).astype(int)
        
    elif threshold_method == 'bayes_optimal':
        # Group-specific threshold optimization (Bayes-optimal approach)
        def objective(params):
            return _cost_sensitive_risk(params)
        
        # Grid search for two thresholds
        thresh_range = np.linspace(0, 1, 21)
        best_risk = float('inf')
        best_thresholds = (0.5, 0.5)
        
        for t0 in thresh_range:
            for t1 in thresh_range:
                risk = _cost_sensitive_risk((t0, t1))
                if risk < best_risk:
                    best_risk = risk
                    best_thresholds = (t0, t1)
        
        optimal_thresholds = {'group_0': best_thresholds[0], 'group_1': best_thresholds[1]}
        
        # Generate predictions with optimal thresholds
        y_pred_constrained = np.zeros_like(y_true)
        y_pred_constrained[group_0_idx] = (y_prob[group_0_idx] >= best_thresholds[0]).astype(int)
        y_pred_constrained[group_1_idx] = (y_prob[group_1_idx] >= best_thresholds[1]).astype(int)
    
    # Compute final metrics
    accuracy_constrained = np.mean(y_true == y_pred_constrained)
    fairness_violation = _compute_fairness_violation(y_pred_constrained)
    cost_sensitive_risk = _cost_sensitive_risk(
        optimal_thresholds['group_0'] if threshold_method == 'single_threshold' 
        else (optimal_thresholds['group_0'], optimal_thresholds['group_1'])
    )
    
    # Compute Bayes-optimal risk (theoretical minimum)
    # This is the risk of the optimal classifier without fairness constraints
    bayes_optimal_risk = 1.0 - accuracy_unconstrained
    
    # Fairness-accuracy ratio
    accuracy_loss = accuracy_unconstrained - accuracy_constrained
    fairness_accuracy_ratio = fairness_violation / max(accuracy_loss, 1e-10)
    
    # Group-specific statistics
    group_statistics = {}
    for group_id, group_mask in [('group_0', group_0_idx), ('group_1', group_1_idx)]:
        group_stats = {
            'size': np.sum(group_mask),
            'base_rate': np.mean(y_true[group_mask]),
            'prediction_rate_unconstrained': np.mean(y_pred_unconstrained[group_mask]),
            'prediction_rate_constrained': np.mean(y_pred_constrained[group_mask]),
            'accuracy_unconstrained': np.mean(y_true[group_mask] == y_pred_unconstrained[group_mask]),
            'accuracy_constrained': np.mean(y_true[group_mask] == y_pred_constrained[group_mask])
        }
        
        # True positive rate and false positive rate
        if np.any(y_true[group_mask] == 1):
            group_stats['tpr_unconstrained'] = np.mean(
                y_pred_unconstrained[group_mask & (y_true == 1)]
            )
            group_stats['tpr_constrained'] = np.mean(
                y_pred_constrained[group_mask & (y_true == 1)]
            )
        
        if np.any(y_true[group_mask] == 0):
            group_stats['fpr_unconstrained'] = np.mean(
                y_pred_unconstrained[group_mask & (y_true == 0)]
            )