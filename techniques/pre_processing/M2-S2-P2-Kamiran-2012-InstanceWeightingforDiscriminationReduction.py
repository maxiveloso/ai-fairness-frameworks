import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional
from sklearn.utils import check_X_y, check_array

def instance_weighting_discrimination_reduction(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    protected_attribute: Union[str, int],
    privileged_groups: Union[list, np.ndarray] = None,
    positive_label: Union[str, int, float] = 1
) -> Dict[str, Any]:
    """
    Compute instance weights for discrimination reduction using Kamiran-Calders method.
    
    This technique assigns weights to training instances to balance the representation
    of different protected groups across positive and negative outcomes. The goal is
    to reduce discrimination by ensuring equal representation of privileged and
    unprivileged groups in both outcome classes.
    
    The method computes four types of weights:
    - w_p_fav: Weight for privileged group with positive outcome
    - w_p_unfav: Weight for privileged group with negative outcome  
    - w_up_fav: Weight for unprivileged group with positive outcome
    - w_up_unfav: Weight for unprivileged group with negative outcome
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or DataFrame
        Feature matrix
    y : array-like of shape (n_samples,)
        Target variable (binary outcomes)
    protected_attribute : str or int
        Column name (if X is DataFrame) or column index (if X is array) 
        of the protected attribute
    privileged_groups : list or array-like, optional
        Values of protected attribute considered privileged. If None,
        assumes binary attribute where 1 is privileged
    positive_label : str, int, or float, default=1
        Value in y considered as positive outcome
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'weights': array of computed weights for each instance
        - 'group_weights': dict mapping (group, outcome) to weight value
        - 'group_counts': dict with counts for each group-outcome combination
        - 'discrimination_before': discrimination measure before weighting
        - 'expected_discrimination_after': expected discrimination after weighting
        - 'weight_statistics': summary statistics of computed weights
        
    References
    ----------
    Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for 
    classification without discrimination. Knowledge and Information Systems, 
    33(1), 1-33.
    """
    
    # Input validation
    if isinstance(X, pd.DataFrame):
        if isinstance(protected_attribute, str):
            if protected_attribute not in X.columns:
                raise ValueError(f"Protected attribute '{protected_attribute}' not found in DataFrame columns")
            protected_values = X[protected_attribute].values
            X_features = X.drop(columns=[protected_attribute]).values
        else:
            raise ValueError("If X is DataFrame, protected_attribute must be column name (str)")
    else:
        X = check_array(X)
        if isinstance(protected_attribute, int):
            if protected_attribute >= X.shape[1]:
                raise ValueError(f"Protected attribute index {protected_attribute} out of bounds")
            protected_values = X[:, protected_attribute]
            X_features = np.delete(X, protected_attribute, axis=1)
        else:
            raise ValueError("If X is array, protected_attribute must be column index (int)")
    
    y = check_array(y, ensure_2d=False)
    
    if len(protected_values) != len(y):
        raise ValueError("X and y must have same number of samples")
    
    # Set privileged groups
    if privileged_groups is None:
        unique_groups = np.unique(protected_values)
        if len(unique_groups) == 2:
            privileged_groups = [max(unique_groups)]  # Assume higher value is privileged
        else:
            raise ValueError("privileged_groups must be specified for non-binary protected attributes")
    
    privileged_groups = np.array(privileged_groups)
    
    # Create binary privileged indicator
    is_privileged = np.isin(protected_values, privileged_groups)
    is_positive = (y == positive_label)
    
    # Count instances in each group-outcome combination
    n = len(y)  # Total instances
    n_fav = np.sum(is_privileged)  # Privileged group
    n_unfav = n - n_fav  # Unprivileged group
    n_p = np.sum(is_positive)  # Positive outcomes
    n_up = n - n_p  # Negative outcomes
    
    # Group-outcome combinations
    n_p_fav = np.sum(is_privileged & is_positive)  # Privileged + Positive
    n_p_unfav = np.sum(~is_privileged & is_positive)  # Unprivileged + Positive
    n_up_fav = np.sum(is_privileged & ~is_positive)  # Privileged + Negative
    n_up_unfav = np.sum(~is_privileged & ~is_positive)  # Unprivileged + Negative
    
    # Validate counts
    if n_p_fav == 0 or n_p_unfav == 0 or n_up_fav == 0 or n_up_unfav == 0:
        raise ValueError("All group-outcome combinations must have at least one instance")
    
    # Compute weights using Kamiran-Calders formulas
    # These weights balance representation across groups and outcomes
    w_p_fav = (n_fav * n_p) / (n * n_p_fav)      # Privileged + Positive
    w_p_unfav = (n_unfav * n_p) / (n * n_p_unfav)  # Unprivileged + Positive  
    w_up_fav = (n_fav * n_up) / (n * n_up_fav)    # Privileged + Negative
    w_up_unfav = (n_unfav * n_up) / (n * n_up_unfav)  # Unprivileged + Negative
    
    # Assign weights to each instance
    weights = np.zeros(n)
    
    # Privileged group with positive outcome
    mask_p_fav = is_privileged & is_positive
    weights[mask_p_fav] = w_p_fav
    
    # Unprivileged group with positive outcome
    mask_p_unfav = ~is_privileged & is_positive
    weights[mask_p_unfav] = w_p_unfav
    
    # Privileged group with negative outcome
    mask_up_fav = is_privileged & ~is_positive
    weights[mask_up_fav] = w_up_fav
    
    # Unprivileged group with negative outcome
    mask_up_unfav = ~is_privileged & ~is_positive
    weights[mask_up_unfav] = w_up_unfav
    
    # Calculate discrimination measures
    # Discrimination = P(Y=1|S=privileged) - P(Y=1|S=unprivileged)
    p_pos_given_priv = n_p_fav / n_fav if n_fav > 0 else 0
    p_pos_given_unpriv = n_p_unfav / n_unfav if n_unfav > 0 else 0
    discrimination_before = p_pos_given_priv - p_pos_given_unpriv
    
    # After weighting, expected discrimination should be reduced
    # Weighted positive rates
    weighted_pos_priv = np.sum(weights[mask_p_fav])
    weighted_total_priv = np.sum(weights[is_privileged])
    weighted_pos_unpriv = np.sum(weights[mask_p_unfav])
    weighted_total_unpriv = np.sum(weights[~is_privileged])
    
    expected_p_pos_priv = weighted_pos_priv / weighted_total_priv if weighted_total_priv > 0 else 0
    expected_p_pos_unpriv = weighted_pos_unpriv / weighted_total_unpriv if weighted_total_unpriv > 0 else 0
    expected_discrimination_after = expected_p_pos_priv - expected_p_pos_unpriv
    
    # Weight statistics
    weight_stats = {
        'mean': np.mean(weights),
        'std': np.std(weights),
        'min': np.min(weights),
        'max': np.max(weights),
        'median': np.median(weights)
    }
    
    return {
        'weights': weights,
        'group_weights': {
            ('privileged', 'positive'): w_p_fav,
            ('unprivileged', 'positive'): w_p_unfav,
            ('privileged', 'negative'): w_up_fav,
            ('unprivileged', 'negative'): w_up_unfav
        },
        'group_counts': {
            'total': n,
            'privileged': n_fav,
            'unprivileged': n_unfav,
            'positive': n_p,
            'negative': n_up,
            ('privileged', 'positive'): n_p_fav,
            ('unprivileged', 'positive'): n_p_unfav,
            ('privileged', 'negative'): n_up_fav,
            ('unprivileged', 'negative'): n_up_unfav
        },
        'discrimination_before': discrimination_before,
        'expected_discrimination_after': expected_discrimination_after,
        'discrimination_reduction': discrimination_before - expected_discrimination_after,
        'weight_statistics': weight_stats
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Create synthetic dataset with bias
    n_samples = 1000
    
    # Protected attribute: 0 = unprivileged, 1 = privileged
    protected_attr = np.random.binomial(1, 0.6, n_samples)  # 60% privileged
    
    # Other features
    X_other = np.random.randn(n_samples, 3)
    
    # Biased outcome: privileged group has higher probability of positive outcome
    prob_positive = np.where(protected_attr == 1, 0.7, 0.3)  # 70% vs 30%
    y = np.random.binomial(1, prob_positive, n_samples)
    
    # Combine features
    X = np.column_stack([protected_attr, X_other])
    
    print("Instance Weighting for Discrimination Reduction Example")
    print("=" * 55)
    
    # Apply instance weighting
    results = instance_weighting_discrimination_reduction(
        X=X, 
        y=y, 
        protected_attribute=0,  # First column is protected attribute
        privileged_groups=[1],
        positive_label=1
    )
    
    print(f"Original discrimination: {results['discrimination_before']:.4f}")
    print(f"Expected discrimination after weighting: {results['expected_discrimination_after']:.4f}")
    print(f"Discrimination reduction: {results['discrimination_reduction']:.4f}")
    print()
    
    print("Group counts:")
    for key, value in results['group_counts'].items():
        if isinstance(key, tuple):
            print(f"  {key[0]} + {key[1]}: {value}")
        else:
            print(f"  {key}: {value}")
    print()
    
    print("Group weights:")
    for key, value in results['group_weights'].items():
        print(f"  {key[0]} + {key[1]}: {value:.4f}")