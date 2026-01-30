import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional
from sklearn.utils import check_X_y, check_array

def inverse_propensity_weighting_for_fairness(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    protected_attribute: Union[str, int],
    return_weights_only: bool = False
) -> Dict[str, Any]:
    """
    Compute inverse propensity weights to mitigate bias in datasets with protected attributes.
    
    This technique assigns weights inversely proportional to the representation of 
    protected attribute-outcome combinations to balance representation and counteract
    historical discrimination in training data.
    
    The algorithm works by:
    1. Calculating frequency of each (protected_attribute, outcome) combination
    2. Computing weights as inverse of these frequencies 
    3. Normalizing weights to maintain original dataset size
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or DataFrame
        Feature matrix containing the protected attribute
    y : array-like of shape (n_samples,)
        Target variable (binary outcomes)
    protected_attribute : str or int
        Column name (if DataFrame) or column index (if array) of protected attribute
    return_weights_only : bool, default=False
        If True, return only the computed weights
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'weights': array of computed inverse propensity weights
        - 'group_frequencies': frequencies of each (protected_attr, outcome) combination
        - 'group_weights': raw inverse frequencies before normalization
        - 'balance_metrics': statistical measures of dataset balance
        
    Raises
    ------
    ValueError
        If protected attribute values are not binary or if inputs are invalid
        
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> 
    >>> # Create sample biased dataset
    >>> np.random.seed(42)
    >>> n_samples = 1000
    >>> protected_attr = np.random.binomial(1, 0.3, n_samples)  # 30% minority group
    >>> # Introduce bias: minority group has lower positive outcome rate
    >>> y = np.where(protected_attr == 1, 
    ...               np.random.binomial(1, 0.3, n_samples),  # 30% positive for minority
    ...               np.random.binomial(1, 0.7, n_samples))  # 70% positive for majority
    >>> X = np.column_stack([protected_attr, np.random.randn(n_samples, 2)])
    >>> 
    >>> result = inverse_propensity_weighting_for_fairness(X, y, protected_attribute=0)
    >>> print(f"Original bias: {np.mean(y[protected_attr==1]) - np.mean(y[protected_attr==0]):.3f}")
    >>> weights = result['weights']
    >>> print(f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    """
    
    # Input validation
    if isinstance(X, pd.DataFrame):
        if isinstance(protected_attribute, str):
            if protected_attribute not in X.columns:
                raise ValueError(f"Protected attribute '{protected_attribute}' not found in DataFrame columns")
            protected_values = X[protected_attribute].values
            X_array = X.values
        else:
            protected_values = X.iloc[:, protected_attribute].values
            X_array = X.values
    else:
        X_array = check_array(X)
        if isinstance(protected_attribute, int):
            if protected_attribute >= X_array.shape[1]:
                raise ValueError(f"Protected attribute index {protected_attribute} out of bounds")
            protected_values = X_array[:, protected_attribute]
        else:
            raise ValueError("Protected attribute must be column index when X is array")
    
    y_array = check_array(y, ensure_2d=False)
    
    # Ensure we have the same number of samples
    if len(protected_values) != len(y_array):
        raise ValueError("X and y must have the same number of samples")
    
    # Check if protected attribute and outcome are binary
    unique_protected = np.unique(protected_values)
    unique_outcomes = np.unique(y_array)
    
    if len(unique_protected) != 2:
        raise ValueError("Protected attribute must be binary")
    if len(unique_outcomes) != 2:
        raise ValueError("Outcome variable must be binary")
    
    # Calculate frequencies for each (protected_attribute, outcome) combination
    group_frequencies = {}
    n_total = len(y_array)
    
    for prot_val in unique_protected:
        for outcome_val in unique_outcomes:
            mask = (protected_values == prot_val) & (y_array == outcome_val)
            freq = np.sum(mask) / n_total
            group_frequencies[(prot_val, outcome_val)] = freq
    
    # Compute raw inverse propensity weights
    # Weight = 1 / P(protected_attribute = a, outcome = y)
    group_weights = {}
    for key, freq in group_frequencies.items():
        if freq > 0:
            group_weights[key] = 1.0 / freq
        else:
            # Handle zero frequency case by assigning very large weight
            group_weights[key] = n_total
    
    # Assign weights to each sample
    sample_weights = np.zeros(len(y_array))
    for i in range(len(y_array)):
        key = (protected_values[i], y_array[i])
        sample_weights[i] = group_weights[key]
    
    # Normalize weights to maintain original dataset size
    # This ensures the weighted dataset has the same effective sample size
    sample_weights = sample_weights * (n_total / np.sum(sample_weights))
    
    if return_weights_only:
        return {'weights': sample_weights}
    
    # Calculate balance metrics
    balance_metrics = {}
    
    # Original demographic parity difference
    orig_pos_rate_prot_0 = np.mean(y_array[protected_values == unique_protected[0]])
    orig_pos_rate_prot_1 = np.mean(y_array[protected_values == unique_protected[1]])
    balance_metrics['original_demographic_parity_diff'] = abs(orig_pos_rate_prot_1 - orig_pos_rate_prot_0)
    
    # Weighted demographic parity difference
    weights_prot_0 = sample_weights[protected_values == unique_protected[0]]
    weights_prot_1 = sample_weights[protected_values == unique_protected[1]]
    y_prot_0 = y_array[protected_values == unique_protected[0]]
    y_prot_1 = y_array[protected_values == unique_protected[1]]
    
    weighted_pos_rate_prot_0 = np.average(y_prot_0, weights=weights_prot_0)
    weighted_pos_rate_prot_1 = np.average(y_prot_1, weights=weights_prot_1)
    balance_metrics['weighted_demographic_parity_diff'] = abs(weighted_pos_rate_prot_1 - weighted_pos_rate_prot_0)
    
    # Weight statistics
    balance_metrics['weight_mean'] = np.mean(sample_weights)
    balance_metrics['weight_std'] = np.std(sample_weights)
    balance_metrics['weight_min'] = np.min(sample_weights)
    balance_metrics['weight_max'] = np.max(sample_weights)
    
    # Effective sample size (measures information loss due to weighting)
    balance_metrics['effective_sample_size'] = (np.sum(sample_weights) ** 2) / np.sum(sample_weights ** 2)
    balance_metrics['relative_efficiency'] = balance_metrics['effective_sample_size'] / n_total
    
    return {
        'weights': sample_weights,
        'group_frequencies': group_frequencies,
        'group_weights': group_weights,
        'balance_metrics': balance_metrics
    }

if __name__ == "__main__":
    # Example usage with synthetic biased dataset
    np.random.seed(42)
    
    # Create biased dataset where protected attribute correlates with outcome
    n_samples = 1000
    
    # Protected attribute: 0 = majority group (70%), 1 = minority group (30%)
    protected_attr = np.random.binomial(1, 0.3, n_samples)
    
    # Introduce systematic bias: minority group has lower positive outcome rate
    # This simulates historical discrimination in the data
    outcome_probs = np.where(protected_attr == 1, 0.3, 0.7)  # 30% vs 70% positive rate
    y = np.random.binomial(1, outcome_probs)
    
    # Additional features (not used in weighting but would be in real model)
    other_features = np.random.randn(n_samples, 3)
    X = np.column_stack([protected_attr, other_features])
    
    print("=== Inverse Propensity Weighting for Fairness ===")
    print(f"Dataset size: {n_samples}")
    print(f"Protected attribute distribution: {np.bincount(protected_attr)}")
    print(f"Outcome distribution: {np.bincount(y)}")
    
    # Apply inverse propensity weighting
    result = inverse_propensity_weighting_for_fairness(X, y, protected_attribute=0)
    
    print("\n--- Group Frequencies ---")
    for (prot_val, outcome_val), freq in result['group_frequencies'].items():
        print(f"P(protected={prot_val}, outcome={outcome_val}) = {freq:.4f}")
    
    print("\n--- Group Weights ---")
    for (prot_val, outcome_val), weight in result['group_weights'].items():
        print(f"Weight(protected={prot_val}, outcome={outcome_val}) = {weight:.4f}")
    
    print("\n--- Balance Metrics ---")
    metrics = result['balance_metrics']
    print(f"Original demographic parity difference: {metrics['original_demographic_parity_diff']:.4f}")
    print(f"Weighted demographic parity difference: {metrics['weighted_demographic_parity_diff']:.4f}")
    print(f"Bias reduction: {(1 - metrics['weighted_demographic_parity_diff']/metrics['original_demographic_parity_diff'])*100:.1f}%")
    
    print(f"\nWeight statistics:")
    print(f"  Mean: {metrics['weight_mean']:.4f}")
    print(f"  Std:  {metrics['weight_std']:.4f}")
    print(f"  Range: [{metrics['weight_min']:.4f}, {metrics['weight_max']:.4f}]")
    
    print(f"\nEffective sample size: {metrics['effective_sample_size']:.1f}")
    print(f"Relative efficiency: {metrics['relative_efficiency']:.3f}")
    
    # Demonstrate usage with pandas DataFrame
    print("\n--- DataFrame Example ---")
    df = pd.DataFrame({
        'protected_attr': protected_attr,
        'feature1': other_features[:, 0],
        'feature2': other_features[:, 1],
        'feature3': other_features[:, 2]
    })
    
    result_df = inverse_propensity_weighting_for_fairness(
        df, y, protected_attribute='protected_attr'
    )
    
    print(f"DataFrame result - bias reduction: {(1 - result_df['balance_metrics']['weighted_demographic_parity_diff']/result_df['balance_metrics']['original_demographic_parity_diff'])*100:.1