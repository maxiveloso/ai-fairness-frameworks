import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from scipy import stats
import warnings

def calibration_fairness_tradeoff_analysis(
    y_true: Union[np.ndarray, pd.Series, List],
    y_scores: Union[np.ndarray, pd.Series, List],
    groups: Union[np.ndarray, pd.Series, List],
    n_bins: int = 10,
    alpha: float = 0.05,
    return_group_metrics: bool = True
) -> Dict:
    """
    Analyze the trade-off between calibration and fairness metrics as described by Kleinberg et al. (2016).
    
    This function evaluates three key fairness criteria:
    1. Calibration within groups: Expected fraction of positive class matches assigned scores within each group
    2. Balance for negative class: Average scores equal across groups for negative class
    3. Balance for positive class: Average scores equal across groups for positive class
    
    The analysis demonstrates the inherent mathematical impossibility of satisfying all three
    criteria simultaneously except in degenerate cases.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1)
    y_scores : array-like
        Predicted scores/probabilities (should be between 0 and 1)
    groups : array-like
        Group membership indicators (e.g., protected attributes)
    n_bins : int, default=10
        Number of bins for calibration analysis
    alpha : float, default=0.05
        Significance level for statistical tests
    return_group_metrics : bool, default=True
        Whether to return detailed metrics for each group
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'calibration_within_groups': Dict with calibration metrics for each group
        - 'balance_negative_class': Dict with balance metrics for negative class
        - 'balance_positive_class': Dict with balance metrics for positive class
        - 'overall_calibration_test': Statistical test for overall calibration
        - 'fairness_violations': Summary of which fairness criteria are violated
        - 'trade_off_summary': Overall assessment of trade-offs
    """
    
    # Input validation
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    groups = np.asarray(groups)
    
    if len(y_true) != len(y_scores) or len(y_true) != len(groups):
        raise ValueError("All input arrays must have the same length")
    
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only binary values (0, 1)")
    
    if np.any(y_scores < 0) or np.any(y_scores > 1):
        raise ValueError("y_scores must be between 0 and 1")
    
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be between 0 and 1")
    
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least 2 groups are required for fairness analysis")
    
    results = {}
    
    # 1. Calibration within groups analysis
    calibration_results = {}
    group_calibration_violations = []
    
    for group in unique_groups:
        group_mask = groups == group
        group_y_true = y_true[group_mask]
        group_y_scores = y_scores[group_mask]
        
        if len(group_y_true) == 0:
            continue
            
        # Create bins for calibration analysis
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(group_y_scores, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_calibration = []
        bin_counts = []
        calibration_errors = []
        
        for i in range(n_bins):
            bin_mask = bin_indices == i
            if np.sum(bin_mask) > 0:
                bin_true_rate = np.mean(group_y_true[bin_mask])
                bin_pred_rate = np.mean(group_y_scores[bin_mask])
                bin_count = np.sum(bin_mask)
                
                bin_calibration.append({
                    'bin_index': i,
                    'bin_range': (bin_boundaries[i], bin_boundaries[i+1]),
                    'true_positive_rate': bin_true_rate,
                    'predicted_positive_rate': bin_pred_rate,
                    'calibration_error': abs(bin_true_rate - bin_pred_rate),
                    'count': bin_count
                })
                calibration_errors.append(abs(bin_true_rate - bin_pred_rate))
                bin_counts.append(bin_count)
        
        # Expected Calibration Error (ECE) for this group
        if bin_counts:
            total_count = sum(bin_counts)
            ece = sum(error * count for error, count in zip(calibration_errors, bin_counts)) / total_count
        else:
            ece = np.nan
        
        # Hosmer-Lemeshow test for calibration
        try:
            # Simplified calibration test using chi-square
            observed_pos = []
            expected_pos = []
            for bin_info in bin_calibration:
                if bin_info['count'] > 0:
                    observed_pos.append(bin_info['true_positive_rate'] * bin_info['count'])
                    expected_pos.append(bin_info['predicted_positive_rate'] * bin_info['count'])
            
            if len(observed_pos) > 1:
                chi2_stat = sum((o - e)**2 / (e + 1e-8) for o, e in zip(observed_pos, expected_pos))
                p_value = 1 - stats.chi2.cdf(chi2_stat, len(observed_pos) - 1)
                calibration_test = {'chi2_statistic': chi2_stat, 'p_value': p_value}
                
                if p_value < alpha:
                    group_calibration_violations.append(group)
            else:
                calibration_test = {'chi2_statistic': np.nan, 'p_value': np.nan}
        except:
            calibration_test = {'chi2_statistic': np.nan, 'p_value': np.nan}
        
        calibration_results[group] = {
            'expected_calibration_error': ece,
            'bin_calibration': bin_calibration,
            'calibration_test': calibration_test,
            'is_calibrated': calibration_test['p_value'] >= alpha if not np.isnan(calibration_test['p_value']) else None
        }
    
    results['calibration_within_groups'] = calibration_results
    
    # 2. Balance for negative class (y_true == 0)
    negative_mask = y_true == 0
    negative_scores_by_group = {}
    negative_group_means = []
    negative_group_sizes = []
    
    for group in unique_groups:
        group_negative_mask = negative_mask & (groups == group)
        if np.sum(group_negative_mask) > 0:
            group_negative_scores = y_scores[group_negative_mask]
            negative_scores_by_group[group] = {
                'mean_score': np.mean(group_negative_scores),
                'std_score': np.std(group_negative_scores),
                'count': len(group_negative_scores)
            }
            negative_group_means.append(np.mean(group_negative_scores))
            negative_group_sizes.append(len(group_negative_scores))
    
    # Test for equal means across groups for negative class
    if len(negative_group_means) >= 2:
        # ANOVA test for equal means
        negative_groups_data = []
        for group in unique_groups:
            group_negative_mask = negative_mask & (groups == group)
            if np.sum(group_negative_mask) > 0:
                negative_groups_data.append(y_scores[group_negative_mask])
        
        if len(negative_groups_data) >= 2 and all(len(data) > 0 for data in negative_groups_data):
            try:
                f_stat, p_value = stats.f_oneway(*negative_groups_data)
                negative_balance_test = {'f_statistic': f_stat, 'p_value': p_value}
                negative_balance_violated = p_value < alpha
            except:
                negative_balance_test = {'f_statistic': np.nan, 'p_value': np.nan}
                negative_balance_violated = True
        else:
            negative_balance_test = {'f_statistic': np.nan, 'p_value': np.nan}
            negative_balance_violated = True
    else:
        negative_balance_test = {'f_statistic': np.nan, 'p_value': np.nan}
        negative_balance_violated = True
    
    results['balance_negative_class'] = {
        'group_statistics': negative_scores_by_group,
        'balance_test': negative_balance_test,
        'balance_violated': negative_balance_violated,
        'mean_difference': np.std(negative_group_means) if negative_group_means else np.nan
    }
    
    # 3. Balance for positive class (y_true == 1)
    positive_mask = y_true == 1
    positive_scores_by_group = {}
    positive_group_means = []
    positive_group_sizes = []
    
    for group in unique_groups:
        group_positive_mask = positive_mask & (groups == group)
        if np.sum(group_positive_mask) > 0:
            group_positive_scores = y_scores[group_positive_mask]
            positive_scores_by_group[group] = {
                'mean_score': np.mean(group_positive_scores),
                'std_score': np.std(group_positive_scores),
                'count': len(group_positive_scores)
            }
            positive_group_means.append(np.mean(group_positive_scores))
            positive_group_sizes.append(len(group_positive_scores))
    
    # Test for equal means across groups for positive class
    if len(positive_group_means) >= 2:
        # ANOVA test for equal means
        positive_groups_data = []
        for group in unique_groups:
            group_positive_mask = positive_mask & (groups == group)
            if np.sum(group_positive_mask) > 0:
                positive_groups_data.append(y_scores[group_positive_mask])
        
        if len(positive_groups_data) >= 2 and all(len(data) > 0 for data in positive_groups_data):
            try:
                f_stat, p_value = stats.f_oneway(*positive_groups_data)
                positive_balance_test = {'f_statistic': f_stat, 'p_value': p_value}
                positive_balance_violated = p_value < alpha
            except:
                positive_balance_test = {'f_statistic': np.nan, 'p_value': np.nan}
                positive_balance_violated = True
        else:
            positive_balance_test = {'f_statistic': np.nan, 'p_value': np.nan}
            positive_balance_violated = True
    else:
        positive_balance_test = {'f_statistic': np.nan, 'p