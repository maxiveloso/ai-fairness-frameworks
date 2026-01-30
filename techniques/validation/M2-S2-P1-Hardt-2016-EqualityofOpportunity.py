import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, Tuple
from sklearn.metrics import roc_curve
from scipy import stats
import warnings

def equality_of_opportunity(y_true: Union[np.ndarray, pd.Series], 
                          y_pred: Union[np.ndarray, pd.Series], 
                          protected_attr: Union[np.ndarray, pd.Series],
                          threshold: Optional[float] = None,
                          optimize_threshold: bool = False,
                          privileged_group: Optional[Union[str, int]] = None) -> Dict[str, Any]:
    """
    Calculate Equality of Opportunity fairness metric for binary classification.
    
    Equality of Opportunity requires that the True Positive Rate (TPR) is equal
    across different groups defined by a protected attribute. This ensures that
    qualified individuals from all groups have equal opportunity to receive
    positive predictions.
    
    The metric measures the difference in TPR between groups:
    EOD = TPR_privileged - TPR_unprivileged
    
    Where TPR = P(ŷ=1|y=1) for each group.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities or binary predictions
    protected_attr : array-like of shape (n_samples,)
        Protected attribute values (e.g., race, gender)
    threshold : float, optional
        Decision threshold for converting probabilities to binary predictions.
        If None and y_pred contains probabilities, uses 0.5
    optimize_threshold : bool, default=False
        Whether to find optimal threshold that minimizes EOD
    privileged_group : str or int, optional
        Value indicating the privileged group. If None, uses the most frequent group
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'equality_of_opportunity_difference': EOD value
        - 'tpr_by_group': TPR for each group
        - 'group_counts': Sample sizes by group
        - 'statistical_parity': Overall fairness assessment
        - 'optimal_threshold': If optimize_threshold=True
        - 'chi2_statistic': Chi-square test statistic
        - 'chi2_pvalue': P-value for independence test
    """
    
    # Input validation
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    protected_attr = np.asarray(protected_attr)
    
    if len(y_true) != len(y_pred) or len(y_true) != len(protected_attr):
        raise ValueError("All input arrays must have the same length")
    
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only binary values (0, 1)")
    
    if len(np.unique(protected_attr)) < 2:
        raise ValueError("protected_attr must contain at least 2 different groups")
    
    # Handle threshold for probability predictions
    if threshold is None:
        if np.any((y_pred > 1) | (y_pred < 0)):
            threshold = 0.5  # Assume probabilities if values are outside [0,1]
        else:
            # Check if predictions are already binary
            unique_preds = np.unique(y_pred)
            if len(unique_preds) <= 2 and np.all(np.isin(unique_preds, [0, 1])):
                threshold = 0.5
                y_pred_binary = y_pred.astype(int)
            else:
                threshold = 0.5
                y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Identify privileged group
    groups = np.unique(protected_attr)
    if privileged_group is None:
        # Use most frequent group as privileged
        group_counts = pd.Series(protected_attr).value_counts()
        privileged_group = group_counts.index[0]
    
    if privileged_group not in groups:
        raise ValueError(f"privileged_group {privileged_group} not found in protected_attr")
    
    # Calculate TPR for each group
    tpr_by_group = {}
    group_counts = {}
    
    for group in groups:
        group_mask = protected_attr == group
        group_true = y_true[group_mask]
        group_pred = y_pred_binary[group_mask]
        
        # Calculate TPR = TP / (TP + FN) = TP / P
        positive_cases = group_true == 1
        if np.sum(positive_cases) == 0:
            warnings.warn(f"No positive cases in group {group}")
            tpr_by_group[group] = 0.0
        else:
            true_positives = np.sum((group_true == 1) & (group_pred == 1))
            total_positives = np.sum(positive_cases)
            tpr_by_group[group] = true_positives / total_positives
        
        group_counts[group] = len(group_true)
    
    # Calculate Equality of Opportunity Difference
    privileged_tpr = tpr_by_group[privileged_group]
    
    # Calculate EOD for all group pairs with privileged group
    eod_values = {}
    for group in groups:
        if group != privileged_group:
            eod_values[f"{privileged_group}_vs_{group}"] = privileged_tpr - tpr_by_group[group]
    
    # Overall EOD (max absolute difference)
    if len(eod_values) > 0:
        overall_eod = max(abs(v) for v in eod_values.values())
    else:
        overall_eod = 0.0
    
    # Statistical significance test using chi-square
    # Test independence between group membership and (y_true, y_pred) pairs
    contingency_data = []
    for group in groups:
        group_mask = protected_attr == group
        group_true = y_true[group_mask]
        group_pred = y_pred_binary[group_mask]
        
        tp = np.sum((group_true == 1) & (group_pred == 1))
        fn = np.sum((group_true == 1) & (group_pred == 0))
        contingency_data.append([tp, fn])
    
    contingency_table = np.array(contingency_data)
    
    # Perform chi-square test if we have enough data
    if contingency_table.shape[0] >= 2 and np.all(contingency_table.sum(axis=1) > 0):
        chi2_stat, chi2_pval = stats.chi2_contingency(contingency_table)[:2]
    else:
        chi2_stat, chi2_pval = np.nan, np.nan
    
    # Threshold optimization if requested
    optimal_threshold = None
    if optimize_threshold and not np.all(np.isin(y_pred, [0, 1])):
        optimal_threshold = _optimize_threshold_for_eod(y_true, y_pred, protected_attr, privileged_group)
    
    # Assess statistical parity
    eod_threshold = 0.1  # Common fairness threshold
    is_fair = overall_eod <= eod_threshold
    
    results = {
        'equality_of_opportunity_difference': overall_eod,
        'eod_by_group_pairs': eod_values,
        'tpr_by_group': tpr_by_group,
        'group_counts': group_counts,
        'threshold_used': threshold,
        'statistical_parity': is_fair,
        'chi2_statistic': chi2_stat,
        'chi2_pvalue': chi2_pval,
        'privileged_group': privileged_group
    }
    
    if optimal_threshold is not None:
        results['optimal_threshold'] = optimal_threshold
    
    return results

def _optimize_threshold_for_eod(y_true: np.ndarray, 
                               y_pred: np.ndarray, 
                               protected_attr: np.ndarray,
                               privileged_group: Union[str, int]) -> float:
    """
    Find optimal threshold that minimizes Equality of Opportunity Difference.
    
    Uses ROC curve analysis to find the threshold that results in the smallest
    difference in True Positive Rates between groups.
    """
    
    # Get unique thresholds from ROC curves of each group
    thresholds_to_test = set()
    groups = np.unique(protected_attr)
    
    for group in groups:
        group_mask = protected_attr == group
        if np.sum(group_mask) > 0 and np.sum(y_true[group_mask]) > 0:
            fpr, tpr, thresholds = roc_curve(y_true[group_mask], y_pred[group_mask])
            thresholds_to_test.update(thresholds)
    
    thresholds_to_test = sorted(list(thresholds_to_test))
    
    best_threshold = 0.5
    min_eod = float('inf')
    
    # Test each threshold
    for threshold in thresholds_to_test:
        if np.isnan(threshold) or np.isinf(threshold):
            continue
            
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Calculate TPR for each group
        tpr_values = []
        for group in groups:
            group_mask = protected_attr == group
            group_true = y_true[group_mask]
            group_pred = y_pred_binary[group_mask]
            
            positive_cases = np.sum(group_true == 1)
            if positive_cases > 0:
                true_positives = np.sum((group_true == 1) & (group_pred == 1))
                tpr = true_positives / positive_cases
                tpr_values.append(tpr)
        
        # Calculate EOD as max difference in TPR
        if len(tpr_values) >= 2:
            current_eod = max(tpr_values) - min(tpr_values)
            if current_eod < min_eod:
                min_eod = current_eod
                best_threshold = threshold
    
    return best_threshold

if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic dataset
    n_samples = 1000
    
    # Protected attribute (0: unprivileged, 1: privileged)
    protected_attr = np.random.binomial(1, 0.6, n_samples)
    
    # Generate features with bias
    X = np.random.normal(0, 1, (n_samples, 2))
    X[protected_attr == 1] += 0.5  # Privileged group has higher feature values
    
    # True labels with some correlation to features
    y_true = (X[:, 0] + X[:, 1] + np.random.normal(0, 0.5, n_samples) > 0).astype(int)
    
    # Biased predictions that favor privileged group
    bias_term = protected_attr * 0.3
    y_pred_proba = 1 / (1 + np.exp(-(X[:, 0] + X[:, 1] + bias_term + np.random.normal(0, 0.2, n_samples))))
    
    # Test equality of opportunity
    results = equality_of_opportunity(
        y_true=y_true,
        y_pred=