import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize_scalar
import warnings

def fairness_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    protected_attribute: np.ndarray,
    method: str = 'equalized_odds',
    alpha: Optional[float] = None,
    n_bins: int = 10,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Implement fairness calibration using post-processing with randomization.
    
    This function implements the fairness calibration technique from Pleiss et al. (2017)
    that adjusts classifier outputs to achieve fairness while maintaining calibration.
    The method uses post-processing with randomization rates for protected groups.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_prob : np.ndarray
        Predicted probabilities from classifier
    protected_attribute : np.ndarray
        Binary protected attribute (0 for non-protected, 1 for protected group)
    method : str, default='equalized_odds'
        Fairness criterion to optimize ('equalized_odds' or 'demographic_parity')
    alpha : float, optional
        Randomization rate. If None, will be optimized automatically
    n_bins : int, default=10
        Number of bins for calibration assessment
    tolerance : float, default=1e-6
        Convergence tolerance for optimization
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'calibrated_probabilities': Adjusted probabilities after fairness calibration
        - 'optimal_alpha': Optimal randomization rate used
        - 'original_metrics': Fairness metrics before calibration
        - 'calibrated_metrics': Fairness metrics after calibration
        - 'calibration_error_original': Original calibration error
        - 'calibration_error_calibrated': Calibration error after adjustment
        - 'group_statistics': Per-group statistics
    """
    
    # Input validation
    y_true = check_array(y_true, ensure_2d=False)
    y_prob = check_array(y_prob, ensure_2d=False)
    protected_attribute = check_array(protected_attribute, ensure_2d=False)
    
    if len(y_true) != len(y_prob) or len(y_true) != len(protected_attribute):
        raise ValueError("All input arrays must have the same length")
    
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only binary values (0, 1)")
    
    if not np.all((y_prob >= 0) & (y_prob <= 1)):
        raise ValueError("y_prob must contain probabilities between 0 and 1")
    
    if not np.all(np.isin(protected_attribute, [0, 1])):
        raise ValueError("protected_attribute must contain only binary values (0, 1)")
    
    if method not in ['equalized_odds', 'demographic_parity']:
        raise ValueError("method must be 'equalized_odds' or 'demographic_parity'")
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    protected_attribute = np.array(protected_attribute)
    
    # Separate groups
    group_0_mask = protected_attribute == 0  # Non-protected group
    group_1_mask = protected_attribute == 1  # Protected group
    
    # Calculate original metrics
    original_metrics = _calculate_fairness_metrics(y_true, y_prob, protected_attribute)
    original_calibration_error = _calculate_calibration_error(y_true, y_prob, n_bins)
    
    # Optimize randomization rate if not provided
    if alpha is None:
        if method == 'equalized_odds':
            alpha = _optimize_equalized_odds_alpha(y_true, y_prob, protected_attribute, tolerance)
        else:  # demographic_parity
            alpha = _optimize_demographic_parity_alpha(y_true, y_prob, protected_attribute, tolerance)
    
    # Apply calibration with randomization
    calibrated_probs = _apply_fairness_calibration(
        y_true, y_prob, protected_attribute, alpha, method
    )
    
    # Calculate calibrated metrics
    calibrated_metrics = _calculate_fairness_metrics(y_true, calibrated_probs, protected_attribute)
    calibrated_calibration_error = _calculate_calibration_error(y_true, calibrated_probs, n_bins)
    
    # Calculate group statistics
    group_stats = _calculate_group_statistics(y_true, y_prob, calibrated_probs, protected_attribute)
    
    return {
        'calibrated_probabilities': calibrated_probs,
        'optimal_alpha': alpha,
        'original_metrics': original_metrics,
        'calibrated_metrics': calibrated_metrics,
        'calibration_error_original': original_calibration_error,
        'calibration_error_calibrated': calibrated_calibration_error,
        'group_statistics': group_stats
    }

def _calculate_fairness_metrics(y_true: np.ndarray, y_prob: np.ndarray, protected_attr: np.ndarray) -> Dict[str, float]:
    """Calculate various fairness metrics."""
    # Convert probabilities to binary predictions using 0.5 threshold
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Separate groups
    group_0_mask = protected_attr == 0
    group_1_mask = protected_attr == 1
    
    # Calculate metrics for each group
    def group_metrics(mask):
        if np.sum(mask) == 0:
            return {'tpr': 0, 'fpr': 0, 'tnr': 0, 'fnr': 0, 'positive_rate': 0}
        
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        positive_rate = np.mean(y_p)  # Positive prediction rate
        
        return {'tpr': tpr, 'fpr': fpr, 'tnr': tnr, 'fnr': fnr, 'positive_rate': positive_rate}
    
    metrics_0 = group_metrics(group_0_mask)
    metrics_1 = group_metrics(group_1_mask)
    
    # Calculate fairness disparities
    equalized_odds_diff = abs(metrics_0['tpr'] - metrics_1['tpr']) + abs(metrics_0['fpr'] - metrics_1['fpr'])
    demographic_parity_diff = abs(metrics_0['positive_rate'] - metrics_1['positive_rate'])
    
    return {
        'group_0_tpr': metrics_0['tpr'],
        'group_1_tpr': metrics_1['tpr'],
        'group_0_fpr': metrics_0['fpr'],
        'group_1_fpr': metrics_1['fpr'],
        'group_0_positive_rate': metrics_0['positive_rate'],
        'group_1_positive_rate': metrics_1['positive_rate'],
        'equalized_odds_difference': equalized_odds_diff,
        'demographic_parity_difference': demographic_parity_diff
    }

def _calculate_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy and confidence in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def _optimize_equalized_odds_alpha(y_true: np.ndarray, y_prob: np.ndarray, 
                                 protected_attr: np.ndarray, tolerance: float) -> float:
    """Optimize alpha for equalized odds constraint."""
    def objective(alpha):
        calibrated_probs = _apply_fairness_calibration(y_true, y_prob, protected_attr, alpha, 'equalized_odds')
        metrics = _calculate_fairness_metrics(y_true, calibrated_probs, protected_attr)
        return metrics['equalized_odds_difference']
    
    # Search for optimal alpha
    result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    return result.x

def _optimize_demographic_parity_alpha(y_true: np.ndarray, y_prob: np.ndarray,
                                     protected_attr: np.ndarray, tolerance: float) -> float:
    """Optimize alpha for demographic parity constraint."""
    def objective(alpha):
        calibrated_probs = _apply_fairness_calibration(y_true, y_prob, protected_attr, alpha, 'demographic_parity')
        metrics = _calculate_fairness_metrics(y_true, calibrated_probs, protected_attr)
        return metrics['demographic_parity_difference']
    
    # Search for optimal alpha
    result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    return result.x

def _apply_fairness_calibration(y_true: np.ndarray, y_prob: np.ndarray, 
                              protected_attr: np.ndarray, alpha: float, method: str) -> np.ndarray:
    """Apply fairness calibration using randomization."""
    calibrated_probs = y_prob.copy()
    
    # Separate groups
    group_0_mask = protected_attr == 0
    group_1_mask = protected_attr == 1
    
    if method == 'equalized_odds':
        # Apply different randomization based on true labels
        # For positive examples in protected group
        pos_protected_mask = group_1_mask & (y_true == 1)
        calibrated_probs[pos_protected_mask] = (1 - alpha) * y_prob[pos_protected_mask] + alpha * 0.5
        
        # For negative examples in protected group
        neg_protected_mask = group_1_mask & (y