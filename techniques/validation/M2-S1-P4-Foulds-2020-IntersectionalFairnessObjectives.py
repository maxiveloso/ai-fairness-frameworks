import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from scipy import stats
from scipy.special import gammaln
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array
import warnings

def intersectional_fairness_objectives(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attributes: Union[np.ndarray, pd.DataFrame],
    attribute_names: Optional[List[str]] = None,
    alpha_smoothing: float = 1.0,
    epsilon: float = 1.0,
    fairness_metric: str = 'demographic_parity',
    min_group_size: int = 10,
    confidence_level: float = 0.95,
    use_bayesian_smoothing: bool = True,
    return_group_metrics: bool = False
) -> Dict[str, Any]:
    """
    Compute intersectional fairness objectives using differential privacy-inspired framework.
    
    This implementation measures fairness across intersectional demographic groups using
    Bayesian modeling approaches to address data sparsity in multi-dimensional protected
    attribute spaces. Uses Dirichlet smoothing for empirical counts.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred : np.ndarray
        Predicted binary labels (0 or 1) or prediction probabilities
    protected_attributes : np.ndarray or pd.DataFrame
        Protected attributes defining demographic groups. Each column represents
        a different protected attribute (e.g., race, gender, age group)
    attribute_names : List[str], optional
        Names of protected attributes. If None, uses default names
    alpha_smoothing : float, default=1.0
        Dirichlet smoothing parameter for handling sparse groups
    epsilon : float, default=1.0
        Privacy parameter for differential privacy framework
    fairness_metric : str, default='demographic_parity'
        Fairness metric to compute. Options: 'demographic_parity', 'equalized_odds',
        'equal_opportunity', 'predictive_parity'
    min_group_size : int, default=10
        Minimum group size to include in analysis
    confidence_level : float, default=0.95
        Confidence level for credible intervals
    use_bayesian_smoothing : bool, default=True
        Whether to use Bayesian smoothing for sparse groups
    return_group_metrics : bool, default=False
        Whether to return individual group metrics
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'overall_fairness_score': Overall intersectional fairness score
        - 'max_disparity': Maximum disparity across all group pairs
        - 'mean_disparity': Mean disparity across all group pairs
        - 'num_intersections': Number of intersectional groups
        - 'valid_groups': Number of groups meeting minimum size requirement
        - 'fairness_violations': Number of significant fairness violations
        - 'group_disparities': Pairwise disparities between groups
        - 'credible_intervals': Bayesian credible intervals for group metrics
        - 'privacy_cost': Estimated privacy cost under differential privacy
        - 'group_metrics': Individual group metrics (if requested)
    """
    
    # Input validation
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    if isinstance(protected_attributes, pd.DataFrame):
        protected_attributes = protected_attributes.values
    protected_attributes = check_array(protected_attributes, ensure_2d=True)
    
    if len(y_true) != len(protected_attributes):
        raise ValueError("y_true and protected_attributes must have same length")
    
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only binary values (0, 1)")
    
    if fairness_metric not in ['demographic_parity', 'equalized_odds', 'equal_opportunity', 'predictive_parity']:
        raise ValueError("Invalid fairness_metric")
    
    if alpha_smoothing <= 0:
        raise ValueError("alpha_smoothing must be positive")
    
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    
    n_samples, n_attributes = protected_attributes.shape
    
    if attribute_names is None:
        attribute_names = [f"attr_{i}" for i in range(n_attributes)]
    
    # Convert predictions to binary if probabilities
    if np.any((y_pred > 0) & (y_pred < 1)):
        y_pred_binary = (y_pred > 0.5).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)
    
    # Create intersectional groups by combining all protected attributes
    # Each unique combination of attribute values forms an intersectional group
    group_labels = []
    for i in range(n_samples):
        group_label = tuple(protected_attributes[i])
        group_labels.append(group_label)
    
    unique_groups = list(set(group_labels))
    group_indices = {group: i for i, group in enumerate(unique_groups)}
    
    # Compute group-specific metrics
    group_metrics = {}
    valid_groups = []
    
    for group in unique_groups:
        mask = np.array([g == group for g in group_labels])
        group_size = np.sum(mask)
        
        if group_size < min_group_size:
            continue
            
        valid_groups.append(group)
        y_true_group = y_true[mask]
        y_pred_group = y_pred_binary[mask]
        
        # Compute fairness metric for this group
        if fairness_metric == 'demographic_parity':
            # P(Y_hat = 1 | A = a)
            metric_value = np.mean(y_pred_group)
        elif fairness_metric == 'equalized_odds':
            # Average of TPR and FPR
            if np.sum(y_true_group == 1) > 0:
                tpr = np.mean(y_pred_group[y_true_group == 1])
            else:
                tpr = 0.0
            if np.sum(y_true_group == 0) > 0:
                fpr = np.mean(y_pred_group[y_true_group == 0])
            else:
                fpr = 0.0
            metric_value = (tpr + (1 - fpr)) / 2
        elif fairness_metric == 'equal_opportunity':
            # TPR: P(Y_hat = 1 | Y = 1, A = a)
            if np.sum(y_true_group == 1) > 0:
                metric_value = np.mean(y_pred_group[y_true_group == 1])
            else:
                metric_value = 0.0
        elif fairness_metric == 'predictive_parity':
            # PPV: P(Y = 1 | Y_hat = 1, A = a)
            if np.sum(y_pred_group == 1) > 0:
                metric_value = np.mean(y_true_group[y_pred_group == 1])
            else:
                metric_value = 0.0
        
        # Apply Bayesian smoothing if enabled
        if use_bayesian_smoothing:
            # Use Beta-Binomial conjugate prior for binary outcomes
            successes = np.sum(y_pred_group) if fairness_metric == 'demographic_parity' else np.sum(y_true_group[y_pred_group == 1]) if fairness_metric == 'predictive_parity' else np.sum(y_pred_group[y_true_group == 1])
            trials = len(y_pred_group) if fairness_metric in ['demographic_parity'] else np.sum(y_pred_group == 1) if fairness_metric == 'predictive_parity' else np.sum(y_true_group == 1)
            
            if trials > 0:
                # Posterior parameters
                alpha_post = alpha_smoothing + successes
                beta_post = alpha_smoothing + trials - successes
                
                # Posterior mean
                metric_value_smoothed = alpha_post / (alpha_post + beta_post)
                
                # Credible interval
                alpha_ci = (1 - confidence_level) / 2
                ci_lower = stats.beta.ppf(alpha_ci, alpha_post, beta_post)
                ci_upper = stats.beta.ppf(1 - alpha_ci, alpha_post, beta_post)
            else:
                metric_value_smoothed = metric_value
                ci_lower, ci_upper = 0.0, 1.0
        else:
            metric_value_smoothed = metric_value
            ci_lower, ci_upper = metric_value, metric_value
        
        group_metrics[group] = {
            'raw_metric': metric_value,
            'smoothed_metric': metric_value_smoothed,
            'group_size': group_size,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    if len(valid_groups) < 2:
        warnings.warn("Less than 2 valid groups found. Cannot compute disparities.")
        return {
            'overall_fairness_score': np.nan,
            'max_disparity': np.nan,
            'mean_disparity': np.nan,
            'num_intersections': len(unique_groups),
            'valid_groups': len(valid_groups),
            'fairness_violations': 0,
            'group_disparities': {},
            'credible_intervals': {},
            'privacy_cost': np.nan,
            'group_metrics': group_metrics if return_group_metrics else None
        }
    
    # Compute pairwise disparities between all valid groups
    disparities = []
    group_disparities = {}
    fairness_violations = 0
    
    for i, group1 in enumerate(valid_groups):
        for j, group2 in enumerate(valid_groups):
            if i < j:  # Avoid duplicate pairs
                metric1 = group_metrics[group1]['smoothed_metric']
                metric2 = group_metrics[group2]['smoothed_metric']
                
                # Compute absolute disparity
                disparity = abs(metric1 - metric2)
                disparities.append(disparity)
                
                # Check if disparity is significant based on credible intervals
                ci1_lower = group_metrics[group1]['ci_lower']
                ci1_upper = group_metrics[group1]['ci_upper']
                ci2_lower = group_metrics[group2]['ci_lower']
                ci2_upper = group_metrics[group2]['ci_upper']
                
                # Non-overlapping credible intervals indicate significant difference
                is_significant = (ci1_upper < ci2_lower) or (ci2_upper < ci1_lower)
                if is_significant:
                    fairness_violations += 1
                
                group_disparities[(group1, group2)] = {
                    'disparity': disparity,
                    'is_significant': is_significant,
                    'group1_metric': metric1,
                    'group2_metric': metric2
                }
    
    # Compute overall fairness metrics
    max_disparity = np.max(disparities) if disparities else 0.0
    mean_disparity