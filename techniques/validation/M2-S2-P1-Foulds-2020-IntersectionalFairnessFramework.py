import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any
from scipy import stats
from scipy.special import gammaln
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels


def intersectional_fairness_framework(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attributes: Union[np.ndarray, pd.DataFrame],
    alpha: float = 1.0,
    fairness_threshold: float = 0.1,
    method: str = 'differential',
    smoothing: bool = True,
    return_subgroup_metrics: bool = True
) -> Dict[str, Any]:
    """
    Implement Intersectional Fairness Framework using differential fairness metrics.
    
    This framework measures fairness across intersectional subgroups defined by
    multiple protected attributes using Bayesian probabilistic modeling with
    Dirichlet smoothing. It computes differential fairness by comparing outcome
    distributions across intersectional groups.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred : np.ndarray
        Predicted binary labels (0 or 1) or prediction probabilities
    protected_attributes : np.ndarray or pd.DataFrame
        Multi-dimensional protected attributes defining intersectional groups
    alpha : float, default=1.0
        Dirichlet smoothing parameter (pseudo-count for Bayesian estimation)
    fairness_threshold : float, default=0.1
        Threshold for determining fairness violations
    method : str, default='differential'
        Fairness metric method ('differential', 'demographic_parity', 'equalized_odds')
    smoothing : bool, default=True
        Whether to apply Dirichlet smoothing to empirical counts
    return_subgroup_metrics : bool, default=True
        Whether to return detailed metrics for each intersectional subgroup
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'overall_fairness_score': Overall fairness score (0=unfair, 1=fair)
        - 'differential_fairness': Differential fairness metric
        - 'max_disparity': Maximum disparity between any two subgroups
        - 'fairness_violations': Number of subgroup pairs violating threshold
        - 'subgroup_metrics': Detailed metrics per intersectional subgroup
        - 'pairwise_disparities': Matrix of pairwise disparities
        - 'bayesian_posterior': Posterior estimates with Dirichlet smoothing
        
    References
    ----------
    Foulds, J. R., Islam, R., Keya, K. N., & Pan, S. (2020). An intersectional 
    definition of fairness. In 2020 IEEE 36th International Conference on Data 
    Engineering (ICDE) (pp. 1918-1921). IEEE.
    """
    
    # Input validation
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only binary values (0, 1)")
    
    # Handle probabilistic predictions
    if not np.all(np.isin(y_pred, [0, 1])):
        if np.all((y_pred >= 0) & (y_pred <= 1)):
            y_pred_binary = (y_pred >= 0.5).astype(int)
        else:
            raise ValueError("y_pred must be binary labels or probabilities in [0,1]")
    else:
        y_pred_binary = y_pred.astype(int)
    
    # Convert protected attributes to DataFrame for easier handling
    if isinstance(protected_attributes, np.ndarray):
        if protected_attributes.ndim == 1:
            protected_attributes = protected_attributes.reshape(-1, 1)
        protected_df = pd.DataFrame(protected_attributes, 
                                  columns=[f'attr_{i}' for i in range(protected_attributes.shape[1])])
    else:
        protected_df = protected_attributes.copy()
    
    if len(protected_df) != len(y_true):
        raise ValueError("protected_attributes must have same length as y_true")
    
    if method not in ['differential', 'demographic_parity', 'equalized_odds']:
        raise ValueError("method must be one of: 'differential', 'demographic_parity', 'equalized_odds'")
    
    # Create intersectional groups by combining all protected attributes
    # Each unique combination of attribute values defines a subgroup
    intersectional_groups = protected_df.apply(
        lambda row: '_'.join(row.astype(str)), axis=1
    )
    unique_groups = intersectional_groups.unique()
    n_groups = len(unique_groups)
    
    if n_groups < 2:
        warnings.warn("Only one intersectional group found. Fairness assessment requires multiple groups.")
        return {
            'overall_fairness_score': 1.0,
            'differential_fairness': 0.0,
            'max_disparity': 0.0,
            'fairness_violations': 0,
            'subgroup_metrics': {},
            'pairwise_disparities': np.array([]),
            'bayesian_posterior': {}
        }
    
    # Compute metrics for each intersectional subgroup
    subgroup_metrics = {}
    group_rates = {}
    
    for group in unique_groups:
        mask = intersectional_groups == group
        group_size = np.sum(mask)
        
        if group_size == 0:
            continue
            
        y_true_group = y_true[mask]
        y_pred_group = y_pred_binary[mask]
        
        # Compute empirical counts
        true_positives = np.sum((y_true_group == 1) & (y_pred_group == 1))
        false_positives = np.sum((y_true_group == 0) & (y_pred_group == 1))
        true_negatives = np.sum((y_true_group == 0) & (y_pred_group == 0))
        false_negatives = np.sum((y_true_group == 1) & (y_pred_group == 0))
        
        # Apply Dirichlet smoothing if requested
        if smoothing:
            true_positives += alpha
            false_positives += alpha
            true_negatives += alpha
            false_negatives += alpha
            smoothed_total = group_size + 4 * alpha
        else:
            smoothed_total = group_size
        
        # Compute fairness-relevant rates
        positive_rate = (true_positives + false_positives) / smoothed_total
        
        if method == 'demographic_parity':
            fairness_rate = positive_rate
        elif method == 'equalized_odds':
            # True positive rate (sensitivity)
            tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            # False positive rate (1 - specificity)
            fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
            fairness_rate = (tpr + (1 - fpr)) / 2  # Average of TPR and TNR
        else:  # differential fairness
            fairness_rate = positive_rate
        
        group_rates[group] = fairness_rate
        
        # Store detailed subgroup metrics
        if return_subgroup_metrics:
            subgroup_metrics[group] = {
                'size': int(group_size),
                'positive_rate': float(positive_rate),
                'true_positives': int(true_positives - alpha if smoothing else true_positives),
                'false_positives': int(false_positives - alpha if smoothing else false_positives),
                'true_negatives': int(true_negatives - alpha if smoothing else true_negatives),
                'false_negatives': int(false_negatives - alpha if smoothing else false_negatives),
                'fairness_rate': float(fairness_rate)
            }
    
    # Compute pairwise disparities between all intersectional groups
    groups_list = list(group_rates.keys())
    n_valid_groups = len(groups_list)
    pairwise_disparities = np.zeros((n_valid_groups, n_valid_groups))
    
    for i, group1 in enumerate(groups_list):
        for j, group2 in enumerate(groups_list):
            if i != j:
                # Differential fairness: absolute difference in rates
                disparity = abs(group_rates[group1] - group_rates[group2])
                pairwise_disparities[i, j] = disparity
    
    # Compute overall fairness metrics
    max_disparity = np.max(pairwise_disparities) if n_valid_groups > 1 else 0.0
    
    # Count fairness violations (pairs exceeding threshold)
    fairness_violations = np.sum(pairwise_disparities > fairness_threshold)
    
    # Overall fairness score (1 = perfectly fair, 0 = maximally unfair)
    overall_fairness_score = max(0.0, 1.0 - (max_disparity / 1.0))
    
    # Differential fairness metric (average disparity)
    differential_fairness = np.mean(pairwise_disparities[pairwise_disparities > 0]) if n_valid_groups > 1 else 0.0
    
    # Bayesian posterior estimates with Dirichlet smoothing
    bayesian_posterior = {}
    if smoothing:
        for group in unique_groups:
            mask = intersectional_groups == group
            group_size = np.sum(mask)
            positive_count = np.sum(y_pred_binary[mask])
            
            # Beta posterior parameters (Beta is conjugate prior for Bernoulli)
            posterior_alpha = positive_count + alpha
            posterior_beta = (group_size - positive_count) + alpha
            
            # Posterior mean and variance
            posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
            posterior_var = (posterior_alpha * posterior_beta) / \
                          ((posterior_alpha + posterior_beta)**2 * (posterior_alpha + posterior_beta + 1))
            
            bayesian_posterior[group] = {
                'posterior_mean': float(posterior_mean),
                'posterior_variance': float(posterior_var),
                'credible_interval_95': [
                    float(stats.beta.ppf(0.025, posterior_alpha, posterior_beta)),
                    float(stats.beta.ppf(0.975, posterior_alpha, posterior_beta))
                ]
            }
    
    return {
        'overall_fairness_score': float(overall_fairness_score),
        'differential_fairness': float(differential_fairness),
        'max_disparity': float(max_disparity),
        'fairness_violations': int(fairness_violations),
        'subgroup_metrics': subgroup_metrics,
        'pairwise_disparities': pairwise_disparities,
        'bayesian_posterior': bayesian_posterior,
        'n_intersectional_groups': int(n_valid_groups),
        'fairness_threshold': float(fairness_threshold),
        'method': method
    }


class IntersectionalFairnessClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier