import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, List
from sklearn.calibration import calibration_curve
from scipy import stats
import warnings

def calibration(y_true: Union[np.ndarray, pd.Series, List], 
                y_prob: Union[np.ndarray, pd.Series, List],
                protected_attribute: Union[np.ndarray, pd.Series, List],
                n_bins: int = 10,
                strategy: str = 'uniform',
                pos_label: Union[int, str] = 1) -> Dict[str, Any]:
    """
    Assess calibration fairness across protected groups using Davies and Goel (2018) approach.
    
    Calibration measures whether predicted probabilities match actual outcome rates within
    each protected group. A well-calibrated model should have predicted probabilities
    that align with true positive rates across all demographic groups.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for positive class
    protected_attribute : array-like of shape (n_samples,)
        Protected group membership (e.g., race, gender)
    n_bins : int, default=10
        Number of bins for calibration curve
    strategy : str, default='uniform'
        Strategy for binning: 'uniform' or 'quantile'
    pos_label : int or str, default=1
        Label of positive class
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'overall_calibration': Overall calibration metrics
        - 'group_calibration': Per-group calibration metrics
        - 'calibration_gap': Maximum difference in calibration between groups
        - 'ece_overall': Expected Calibration Error overall
        - 'ece_by_group': Expected Calibration Error by group
        - 'reliability_diagram_data': Data for plotting reliability diagrams
        - 'statistical_tests': Chi-square tests for calibration
    """
    
    # Input validation
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    protected_attribute = np.asarray(protected_attribute)
    
    if len(y_true) != len(y_prob) or len(y_true) != len(protected_attribute):
        raise ValueError("All input arrays must have the same length")
    
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only binary values (0, 1)")
    
    if not np.all((y_prob >= 0) & (y_prob <= 1)):
        raise ValueError("y_prob must contain probabilities between 0 and 1")
    
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    
    if strategy not in ['uniform', 'quantile']:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    
    # Convert to binary if pos_label is specified
    y_binary = (y_true == pos_label).astype(int)
    
    # Get unique groups
    groups = np.unique(protected_attribute)
    
    # Overall calibration
    try:
        fraction_pos_overall, mean_pred_overall = calibration_curve(
            y_binary, y_prob, n_bins=n_bins, strategy=strategy
        )
    except ValueError as e:
        warnings.warn(f"Could not compute overall calibration curve: {e}")
        fraction_pos_overall = np.array([])
        mean_pred_overall = np.array([])
    
    # Group-specific calibration
    group_calibration = {}
    reliability_data = {}
    ece_by_group = {}
    
    for group in groups:
        group_mask = protected_attribute == group
        y_group = y_binary[group_mask]
        prob_group = y_prob[group_mask]
        
        if len(y_group) == 0:
            continue
            
        try:
            fraction_pos_group, mean_pred_group = calibration_curve(
                y_group, prob_group, n_bins=n_bins, strategy=strategy
            )
            
            # Calculate Expected Calibration Error (ECE) for this group
            # ECE is the weighted average of calibration errors across bins
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            total_samples = len(prob_group)
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (prob_group > bin_lower) & (prob_group <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_group[in_bin].mean()
                    avg_confidence_in_bin = prob_group[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            ece_by_group[group] = ece
            
            # Store calibration metrics
            group_calibration[group] = {
                'fraction_positives': fraction_pos_group,
                'mean_predicted_value': mean_pred_group,
                'calibration_error': np.mean(np.abs(fraction_pos_group - mean_pred_group)),
                'max_calibration_error': np.max(np.abs(fraction_pos_group - mean_pred_group)),
                'sample_size': len(y_group)
            }
            
            reliability_data[group] = {
                'mean_predicted_value': mean_pred_group,
                'fraction_positives': fraction_pos_group
            }
            
        except ValueError as e:
            warnings.warn(f"Could not compute calibration for group {group}: {e}")
            group_calibration[group] = {
                'fraction_positives': np.array([]),
                'mean_predicted_value': np.array([]),
                'calibration_error': np.nan,
                'max_calibration_error': np.nan,
                'sample_size': len(y_group)
            }
            ece_by_group[group] = np.nan
    
    # Calculate overall ECE
    ece_overall = 0
    if len(fraction_pos_overall) > 0:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_binary[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece_overall += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    else:
        ece_overall = np.nan
    
    # Calculate calibration gap (maximum difference between groups)
    calibration_errors = [metrics['calibration_error'] for metrics in group_calibration.values() 
                         if not np.isnan(metrics['calibration_error'])]
    
    if len(calibration_errors) >= 2:
        calibration_gap = max(calibration_errors) - min(calibration_errors)
    else:
        calibration_gap = np.nan
    
    # Statistical tests for calibration
    # Hosmer-Lemeshow-like test for each group
    statistical_tests = {}
    
    for group in groups:
        group_mask = protected_attribute == group
        y_group = y_binary[group_mask]
        prob_group = y_prob[group_mask]
        
        if len(y_group) < n_bins:
            statistical_tests[group] = {
                'chi2_statistic': np.nan,
                'p_value': np.nan,
                'test_name': 'Hosmer-Lemeshow-like test'
            }
            continue
        
        # Create bins and calculate expected vs observed
        try:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(prob_group, bin_boundaries) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            observed = []
            expected = []
            
            for i in range(n_bins):
                bin_mask = bin_indices == i
                if np.sum(bin_mask) > 0:
                    obs_pos = np.sum(y_group[bin_mask])
                    exp_pos = np.sum(prob_group[bin_mask])
                    
                    observed.extend([obs_pos, np.sum(bin_mask) - obs_pos])
                    expected.extend([exp_pos, np.sum(bin_mask) - exp_pos])
            
            if len(observed) > 0 and all(e > 0 for e in expected):
                chi2_stat, p_value = stats.chisquare(observed, expected)
                statistical_tests[group] = {
                    'chi2_statistic': chi2_stat,
                    'p_value': p_value,
                    'test_name': 'Hosmer-Lemeshow-like test'
                }
            else:
                statistical_tests[group] = {
                    'chi2_statistic': np.nan,
                    'p_value': np.nan,
                    'test_name': 'Hosmer-Lemeshow-like test'
                }
                
        except Exception as e:
            statistical_tests[group] = {
                'chi2_statistic': np.nan,
                'p_value': np.nan,
                'test_name': 'Hosmer-Lemeshow-like test',
                'error': str(e)
            }
    
    return {
        'overall_calibration': {
            'fraction_positives': fraction_pos_overall,
            'mean_predicted_value': mean_pred_overall,
            'calibration_error': np.mean(np.abs(fraction_pos_overall - mean_pred_overall)) if len(fraction_pos_overall) > 0 else np.nan
        },
        'group_calibration': group_calibration,
        'calibration_gap': calibration_gap,
        'ece_overall': ece_overall,
        'ece_by_group': ece_by_group,
        'reliability_diagram_data': reliability_data,
        'statistical_tests': statistical_tests,
        'n_bins': n_bins,
        'strategy': strategy,
        'groups': list(groups)
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic dataset
    n_samples = 1000
    
    # Create two groups with different calibration properties
    group_a_size = 600
    group_b_size = 400
    
    # Group A: Well-calibrated
    y_true_a = np.random.binomial(1, 0.3, group_a_size)
    y_prob_a = np.random.beta(2, 5, group_a_size)  # Probabilities skewed toward lower values
    # Adjust probabilities to match true rate approximately
    y_prob_a =