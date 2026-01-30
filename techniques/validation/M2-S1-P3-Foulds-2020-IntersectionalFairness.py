import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
import warnings

def intersectional_fairness(
    outcomes: Union[np.ndarray, pd.Series],
    protected_attributes: Union[np.ndarray, pd.DataFrame],
    epsilon: float = 1.0,
    delta: float = 1e-5,
    method: str = 'mutual_info',
    intersectional_groups: Optional[List[List[str]]] = None,
    return_group_metrics: bool = True
) -> Dict[str, Union[float, Dict, List]]:
    """
    Compute intersectional fairness metric using differential privacy principles.
    
    This implementation measures how much information about protected attributes
    can be inferred from outcomes across intersecting demographic groups.
    Lower values indicate better fairness (less information leakage).
    
    The metric is based on differential privacy concepts where we measure
    the mutual information between outcomes and protected attributes,
    ensuring that adversaries cannot learn much about sensitive attributes
    from the model outcomes.
    
    Parameters
    ----------
    outcomes : array-like of shape (n_samples,)
        Binary or continuous outcomes/predictions to evaluate
    protected_attributes : array-like of shape (n_samples, n_attributes)
        Protected attributes (e.g., race, gender, age groups)
    epsilon : float, default=1.0
        Privacy parameter - smaller values require stronger privacy protection
    delta : float, default=1e-5
        Privacy parameter for approximate differential privacy
    method : str, default='mutual_info'
        Method to compute information leakage ('mutual_info' or 'correlation')
    intersectional_groups : list of lists, optional
        Specific intersectional groups to analyze. If None, all combinations considered
    return_group_metrics : bool, default=True
        Whether to return per-group fairness metrics
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'overall_fairness_score': Overall intersectional fairness metric (0-1, lower is better)
        - 'privacy_loss': Estimated privacy loss under (epsilon, delta)-DP
        - 'information_leakage': Raw information leakage score
        - 'group_fairness_scores': Per-group fairness scores (if return_group_metrics=True)
        - 'intersectional_violations': Groups violating fairness thresholds
        - 'differential_privacy_satisfied': Whether DP constraints are satisfied
        - 'method_used': Method used for computation
        
    Raises
    ------
    ValueError
        If inputs have mismatched lengths or invalid parameters
    """
    
    # Input validation
    outcomes = np.asarray(outcomes)
    if isinstance(protected_attributes, pd.DataFrame):
        attr_names = protected_attributes.columns.tolist()
        protected_attributes = protected_attributes.values
    else:
        protected_attributes = np.asarray(protected_attributes)
        if protected_attributes.ndim == 1:
            protected_attributes = protected_attributes.reshape(-1, 1)
        attr_names = [f'attr_{i}' for i in range(protected_attributes.shape[1])]
    
    if len(outcomes) != len(protected_attributes):
        raise ValueError("outcomes and protected_attributes must have same length")
    
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    
    if delta < 0 or delta >= 1:
        raise ValueError("delta must be in [0, 1)")
    
    if method not in ['mutual_info', 'correlation']:
        raise ValueError("method must be 'mutual_info' or 'correlation'")
    
    n_samples, n_attributes = protected_attributes.shape
    
    # Encode categorical variables
    le_outcomes = LabelEncoder()
    outcomes_encoded = le_outcomes.fit_transform(outcomes)
    
    attr_encoders = []
    attrs_encoded = np.zeros_like(protected_attributes, dtype=int)
    for i in range(n_attributes):
        le = LabelEncoder()
        attrs_encoded[:, i] = le.fit_transform(protected_attributes[:, i])
        attr_encoders.append(le)
    
    # Create intersectional groups by combining all protected attributes
    # Each unique combination of attribute values forms an intersectional group
    intersectional_labels = np.zeros(n_samples, dtype=int)
    unique_combinations = {}
    combination_idx = 0
    
    for i in range(n_samples):
        combination = tuple(attrs_encoded[i, :])
        if combination not in unique_combinations:
            unique_combinations[combination] = combination_idx
            combination_idx += 1
        intersectional_labels[i] = unique_combinations[combination]
    
    # Compute information leakage for overall fairness
    if method == 'mutual_info':
        # Mutual information between outcomes and intersectional groups
        overall_leakage = mutual_info_score(outcomes_encoded, intersectional_labels)
        
        # Normalize by maximum possible mutual information
        max_mi = min(stats.entropy(np.bincount(outcomes_encoded) + 1e-10),
                    stats.entropy(np.bincount(intersectional_labels) + 1e-10))
        if max_mi > 0:
            overall_leakage = overall_leakage / max_mi
            
    else:  # correlation method
        # Use correlation coefficient as proxy for information leakage
        if len(np.unique(outcomes_encoded)) > 2:
            # Continuous outcomes - use Pearson correlation
            overall_leakage = abs(np.corrcoef(outcomes_encoded, intersectional_labels)[0, 1])
        else:
            # Binary outcomes - use point-biserial correlation
            overall_leakage = abs(stats.pointbiserialr(outcomes_encoded, intersectional_labels)[0])
        
        if np.isnan(overall_leakage):
            overall_leakage = 0.0
    
    # Compute per-group fairness metrics if requested
    group_fairness_scores = {}
    intersectional_violations = []
    
    if return_group_metrics:
        for combination, group_idx in unique_combinations.items():
            group_mask = intersectional_labels == group_idx
            group_size = np.sum(group_mask)
            
            if group_size < 5:  # Skip very small groups
                continue
                
            group_outcomes = outcomes_encoded[group_mask]
            
            # Compute group-specific information leakage
            if method == 'mutual_info':
                # Compare group outcome distribution to overall distribution
                group_outcome_dist = np.bincount(group_outcomes, minlength=len(np.unique(outcomes_encoded)))
                overall_outcome_dist = np.bincount(outcomes_encoded)
                
                # Normalize distributions
                group_outcome_dist = group_outcome_dist / np.sum(group_outcome_dist)
                overall_outcome_dist = overall_outcome_dist / np.sum(overall_outcome_dist)
                
                # KL divergence as fairness metric
                group_fairness = stats.entropy(group_outcome_dist + 1e-10, overall_outcome_dist + 1e-10)
                
            else:  # correlation method
                # Deviation from overall outcome rate
                group_mean = np.mean(group_outcomes)
                overall_mean = np.mean(outcomes_encoded)
                group_fairness = abs(group_mean - overall_mean) / (overall_mean + 1e-10)
            
            # Create interpretable group name
            group_name = '_'.join([f"{attr_names[i]}={combination[i]}" for i in range(len(combination))])
            group_fairness_scores[group_name] = {
                'fairness_score': group_fairness,
                'group_size': group_size,
                'outcome_rate': np.mean(group_outcomes)
            }
            
            # Check for fairness violations (threshold based on epsilon)
            fairness_threshold = epsilon / 10  # Heuristic threshold
            if group_fairness > fairness_threshold:
                intersectional_violations.append(group_name)
    
    # Estimate privacy loss under differential privacy
    # Privacy loss is related to the sensitivity of the query and epsilon
    sensitivity = 1.0  # Assume unit sensitivity for binary outcomes
    privacy_loss = sensitivity / epsilon
    
    # Check if differential privacy constraints are satisfied
    # This is a simplified check based on information leakage
    dp_threshold = epsilon * sensitivity + np.log(1/delta) / n_samples
    dp_satisfied = overall_leakage <= dp_threshold
    
    # Compute overall fairness score (0-1 scale, lower is better)
    # Combines information leakage with privacy considerations
    overall_fairness_score = min(1.0, overall_leakage + privacy_loss/10)
    
    results = {
        'overall_fairness_score': float(overall_fairness_score),
        'privacy_loss': float(privacy_loss),
        'information_leakage': float(overall_leakage),
        'intersectional_violations': intersectional_violations,
        'differential_privacy_satisfied': bool(dp_satisfied),
        'method_used': method,
        'n_intersectional_groups': len(unique_combinations),
        'epsilon': epsilon,
        'delta': delta
    }
    
    if return_group_metrics:
        results['group_fairness_scores'] = group_fairness_scores
    
    return results

if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic protected attributes
    race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples, 
                           p=[0.6, 0.2, 0.15, 0.05])
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.5, 0.5])
    age_group = np.random.choice(['Young', 'Middle', 'Senior'], n_samples, 
                                p=[0.3, 0.5, 0.2])
    
    # Create biased outcomes that depend on protected attributes
    bias_scores = np.zeros(n_samples)
    for i in range(n_samples):
        if race[i] == 'White':
            bias_scores[i] += 0.3
        if gender[i] == 'Male':
            bias_scores[i] += 0.2
        if age_group[i] == 'Young':
            bias_scores[i] += 0.1
    
    # Add noise and convert to binary outcomes
    outcomes = (bias_scores + np.random.normal(0, 0.3, n_samples)) > 0.2
    
    # Create DataFrame for protected attributes
    protected_attrs = pd.DataFrame({
        'race': race,
        'gender': gender,
        'age_group': age_group
    })
    
    print("Intersectional Fairness Analysis")
    print("=" * 40)
    
    # Test with different epsilon values
    for eps in [0.1, 1.0, 5.0]:
        print(f"\nEpsilon = {eps}")
        print("-" * 20)
        
        results = intersectional_fairness(
            outcomes=outcomes,
            protected_attributes=protected_attrs,
            epsilon=eps,
            method='mutual_info',
            return_group_metrics=True
        )
        
        print(f"Overall Fairness Score: {results['overall_fairness_score']:.4f}")
        print(f"Information Leakage: {results['information_leakage']:.4f}")
        print(f"Privacy Loss: {results['privacy_loss']:.4f}")
        print(f"Intersectional Groups: {results['n_intersect