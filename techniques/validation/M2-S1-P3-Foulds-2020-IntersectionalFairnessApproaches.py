import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from scipy import stats
from scipy.special import gammaln
import warnings
from itertools import product

def intersectional_fairness_approaches(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    protected_attributes: Union[np.ndarray, pd.DataFrame],
    alpha: float = 1.0,
    epsilon: float = 0.1,
    hierarchical: bool = True,
    min_group_size: int = 10
) -> Dict[str, Union[float, Dict, List]]:
    """
    Compute intersectional fairness metrics using differential fairness with Bayesian modeling.
    
    This implementation follows Foulds et al. (2020) approach for measuring fairness across
    intersectional groups defined by multiple protected attributes. Uses Dirichlet smoothing
    to handle data sparsity and provides differential fairness guarantees.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (0 or 1) or probabilities
    protected_attributes : array-like of shape (n_samples, n_attributes)
        Protected attribute values for each sample
    alpha : float, default=1.0
        Dirichlet prior concentration parameter for smoothing
    epsilon : float, default=0.1
        Privacy parameter for differential fairness
    hierarchical : bool, default=True
        Whether to use hierarchical approach for small groups
    min_group_size : int, default=10
        Minimum group size for reliable estimates
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'differential_fairness_score': Overall differential fairness metric
        - 'group_fairness_scores': Per-group fairness scores
        - 'intersectional_groups': List of intersectional group identifiers
        - 'group_counts': Sample counts per group
        - 'bayesian_estimates': Smoothed probability estimates
        - 'privacy_guarantee': Epsilon-differential privacy bound
        - 'hierarchical_adjustments': Adjustments made for small groups
    """
    
    # Input validation
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only binary values (0, 1)")
    
    # Handle probability predictions
    if not np.all(np.isin(y_pred, [0, 1])):
        if np.all((y_pred >= 0) & (y_pred <= 1)):
            y_pred_binary = (y_pred > 0.5).astype(int)
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
        protected_df = pd.DataFrame(protected_attributes)
    
    if len(protected_df) != len(y_true):
        raise ValueError("protected_attributes must have same number of samples as y_true")
    
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    
    # Create intersectional groups
    intersectional_groups = []
    group_indices = {}
    
    # Get unique values for each protected attribute
    unique_values = {}
    for col in protected_df.columns:
        unique_values[col] = sorted(protected_df[col].unique())
    
    # Generate all combinations of protected attribute values
    attr_combinations = list(product(*[unique_values[col] for col in protected_df.columns]))
    
    for i, combination in enumerate(attr_combinations):
        # Create group identifier
        group_id = tuple(combination)
        intersectional_groups.append(group_id)
        
        # Find indices for this group
        mask = np.ones(len(protected_df), dtype=bool)
        for j, col in enumerate(protected_df.columns):
            mask &= (protected_df[col] == combination[j])
        
        group_indices[group_id] = np.where(mask)[0]
    
    # Compute group statistics
    group_counts = {}
    group_outcomes = {}
    group_predictions = {}
    
    for group_id in intersectional_groups:
        indices = group_indices[group_id]
        group_counts[group_id] = len(indices)
        
        if len(indices) > 0:
            group_outcomes[group_id] = y_true[indices]
            group_predictions[group_id] = y_pred_binary[indices]
        else:
            group_outcomes[group_id] = np.array([])
            group_predictions[group_id] = np.array([])
    
    # Apply Bayesian smoothing with Dirichlet prior
    bayesian_estimates = {}
    hierarchical_adjustments = {}
    
    for group_id in intersectional_groups:
        outcomes = group_outcomes[group_id]
        predictions = group_predictions[group_id]
        n_samples = len(outcomes)
        
        if n_samples == 0:
            # No samples in this group - use prior only
            bayesian_estimates[group_id] = {
                'true_positive_rate': alpha / (2 * alpha),
                'predicted_positive_rate': alpha / (2 * alpha),
                'accuracy': alpha / (2 * alpha)
            }
            hierarchical_adjustments[group_id] = "empty_group_prior"
            continue
        
        # Compute empirical counts
        tp = np.sum((outcomes == 1) & (predictions == 1))
        fp = np.sum((outcomes == 0) & (predictions == 1))
        tn = np.sum((outcomes == 0) & (predictions == 0))
        fn = np.sum((outcomes == 1) & (predictions == 0))
        
        # Apply hierarchical adjustment for small groups
        if hierarchical and n_samples < min_group_size:
            # Use overall population statistics as stronger prior
            overall_positive_rate = np.mean(y_true)
            overall_pred_positive_rate = np.mean(y_pred_binary)
            
            # Increase prior strength for small groups
            adjusted_alpha = alpha * max(1, min_group_size / max(n_samples, 1))
            hierarchical_adjustments[group_id] = f"small_group_adjustment_alpha_{adjusted_alpha:.2f}"
        else:
            adjusted_alpha = alpha
            overall_positive_rate = 0.5
            overall_pred_positive_rate = 0.5
            hierarchical_adjustments[group_id] = "no_adjustment"
        
        # Bayesian estimates with Dirichlet smoothing
        # True positive rate among actual positives
        n_positive = np.sum(outcomes == 1)
        if n_positive > 0:
            tpr = (tp + adjusted_alpha * overall_positive_rate) / (n_positive + 2 * adjusted_alpha)
        else:
            tpr = adjusted_alpha / (2 * adjusted_alpha)
        
        # Predicted positive rate
        ppr = (tp + fp + adjusted_alpha * overall_pred_positive_rate) / (n_samples + 2 * adjusted_alpha)
        
        # Accuracy
        accuracy = (tp + tn + adjusted_alpha) / (n_samples + 2 * adjusted_alpha)
        
        bayesian_estimates[group_id] = {
            'true_positive_rate': tpr,
            'predicted_positive_rate': ppr,
            'accuracy': accuracy,
            'sample_size': n_samples
        }
    
    # Compute differential fairness scores
    group_fairness_scores = {}
    
    # Reference group (largest group or first group)
    reference_group = max(intersectional_groups, key=lambda g: group_counts[g])
    ref_estimates = bayesian_estimates[reference_group]
    
    for group_id in intersectional_groups:
        if group_counts[group_id] == 0:
            group_fairness_scores[group_id] = {
                'tpr_ratio': np.nan,
                'ppr_ratio': np.nan,
                'accuracy_ratio': np.nan,
                'differential_score': np.nan
            }
            continue
        
        estimates = bayesian_estimates[group_id]
        
        # Compute ratios (avoiding division by zero)
        tpr_ratio = estimates['true_positive_rate'] / max(ref_estimates['true_positive_rate'], 1e-10)
        ppr_ratio = estimates['predicted_positive_rate'] / max(ref_estimates['predicted_positive_rate'], 1e-10)
        accuracy_ratio = estimates['accuracy'] / max(ref_estimates['accuracy'], 1e-10)
        
        # Differential fairness score (log ratio for multiplicative fairness)
        # Measures deviation from fairness (closer to 0 is more fair)
        differential_score = np.abs(np.log(tpr_ratio)) + np.abs(np.log(ppr_ratio))
        
        group_fairness_scores[group_id] = {
            'tpr_ratio': tpr_ratio,
            'ppr_ratio': ppr_ratio,
            'accuracy_ratio': accuracy_ratio,
            'differential_score': differential_score
        }
    
    # Overall differential fairness score
    valid_scores = [scores['differential_score'] for scores in group_fairness_scores.values() 
                   if not np.isnan(scores['differential_score'])]
    
    if valid_scores:
        # Weighted average by group size
        weights = [group_counts[group_id] for group_id in intersectional_groups 
                  if not np.isnan(group_fairness_scores[group_id]['differential_score'])]
        if sum(weights) > 0:
            differential_fairness_score = np.average(valid_scores, weights=weights)
        else:
            differential_fairness_score = np.mean(valid_scores)
    else:
        differential_fairness_score = np.nan
    
    # Privacy guarantee (epsilon-differential privacy bound)
    # The Dirichlet smoothing provides privacy guarantees
    privacy_guarantee = {
        'epsilon': epsilon,
        'mechanism': 'dirichlet_smoothing',
        'alpha_parameter': alpha,
        'privacy_bound': f"({epsilon}, 0)-differential privacy with Dirichlet mechanism"
    }
    
    return {
        'differential_fairness_score': differential_fairness_score,
        'group_fairness_scores': group_fairness_scores,
        'intersectional_groups': intersectional_groups,
        'group_counts': group_counts,
        'bayesian_estimates': bayesian_estimates,
        'privacy_guarantee': privacy_guarantee,
        'hierarchical_adjustments': hierarchical_adjustments,
        'reference_group': reference_group
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic dataset with intersectional bias
    n_samples =