import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional
from scipy import stats
import warnings

def disparate_impact_removal(
    X: Union[np.ndarray, pd.DataFrame],
    protected_attribute: Union[str, int, np.ndarray],
    repair_level: float = 1.0,
    features_to_repair: Optional[Union[list, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Apply Disparate Impact Removal to transform feature distributions.
    
    This technique removes disparate impact by transforming feature distributions
    within protected groups to a common distribution using a geometric repair method.
    The algorithm preserves rank-ordering within groups while making feature
    distributions independent of the protected attribute.
    
    The repair process uses a lambda parameter (repair_level) that controls the
    degree of transformation:
    - repair_level = 0.0: no repair (original data)
    - repair_level = 1.0: full repair (complete independence)
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input feature matrix
    protected_attribute : str, int, or array-like
        Protected attribute column name (if X is DataFrame), column index,
        or array of protected attribute values
    repair_level : float, default=1.0
        Repair parameter lambda in [0.0, 1.0] controlling transformation degree
    features_to_repair : list or array-like, optional
        Indices or names of features to repair. If None, repairs all features
        except the protected attribute
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'repaired_data': Transformed feature matrix
        - 'repair_level': Applied repair level
        - 'features_repaired': List of repaired feature indices/names
        - 'group_statistics': Statistics for each protected group
        - 'disparate_impact_before': DI ratios before repair
        - 'disparate_impact_after': DI ratios after repair
        - 'balanced_error_rate': Estimated BER improvement
    """
    
    # Input validation
    if not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise TypeError("X must be numpy array or pandas DataFrame")
    
    if not 0.0 <= repair_level <= 1.0:
        raise ValueError("repair_level must be between 0.0 and 1.0")
    
    # Convert to DataFrame for easier handling
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X)
        feature_names = list(range(X.shape[1]))
    else:
        X_df = X.copy()
        feature_names = list(X_df.columns)
    
    # Extract protected attribute
    if isinstance(protected_attribute, str):
        if protected_attribute not in X_df.columns:
            raise ValueError(f"Protected attribute '{protected_attribute}' not found in DataFrame")
        protected_col = X_df[protected_attribute].values
        protected_idx = protected_attribute
    elif isinstance(protected_attribute, int):
        if protected_attribute >= X_df.shape[1]:
            raise ValueError("Protected attribute index out of bounds")
        protected_col = X_df.iloc[:, protected_attribute].values
        protected_idx = protected_attribute
    else:
        protected_col = np.array(protected_attribute)
        if len(protected_col) != len(X_df):
            raise ValueError("Protected attribute length must match number of samples")
        protected_idx = 'protected_attr'
    
    # Determine features to repair
    if features_to_repair is None:
        if isinstance(protected_idx, str) and protected_idx in feature_names:
            features_to_repair = [f for f in feature_names if f != protected_idx]
        elif isinstance(protected_idx, int):
            features_to_repair = [i for i in range(len(feature_names)) if i != protected_idx]
        else:
            features_to_repair = feature_names
    
    # Get unique groups in protected attribute
    unique_groups = np.unique(protected_col)
    if len(unique_groups) < 2:
        warnings.warn("Protected attribute has less than 2 unique values")
    
    # Initialize results
    X_repaired = X_df.copy()
    group_stats = {}
    di_before = {}
    di_after = {}
    
    # Calculate statistics for each feature to repair
    for feature in features_to_repair:
        if isinstance(feature, str):
            feature_data = X_df[feature].values
            feature_key = feature
        else:
            feature_data = X_df.iloc[:, feature].values
            feature_key = f"feature_{feature}"
        
        # Skip non-numeric features
        if not np.issubdtype(feature_data.dtype, np.number):
            continue
        
        # Calculate group-specific statistics
        group_data = {}
        group_cdfs = {}
        
        for group in unique_groups:
            mask = protected_col == group
            group_values = feature_data[mask]
            
            # Store group statistics
            group_data[group] = {
                'values': group_values,
                'mean': np.mean(group_values),
                'std': np.std(group_values),
                'size': len(group_values),
                'mask': mask
            }
            
            # Calculate empirical CDF for this group
            sorted_vals = np.sort(group_values)
            group_cdfs[group] = (sorted_vals, np.arange(1, len(sorted_vals) + 1) / len(sorted_vals))
        
        group_stats[feature_key] = group_data
        
        # Calculate disparate impact before repair
        # Using mean ratio as a simple DI measure
        group_means = [group_data[g]['mean'] for g in unique_groups]
        di_before[feature_key] = min(group_means) / max(group_means) if max(group_means) != 0 else 1.0
        
        # Apply repair transformation
        if repair_level > 0.0:
            # Create combined distribution for target
            all_values = feature_data
            combined_mean = np.mean(all_values)
            combined_std = np.std(all_values)
            
            # Apply geometric repair to each group
            for group in unique_groups:
                mask = group_data[group]['mask']
                original_values = group_data[group]['values']
                
                # Rank-preserving transformation
                # Step 1: Get ranks within group
                ranks = stats.rankdata(original_values, method='average')
                
                # Step 2: Map ranks to target distribution quantiles
                quantiles = (ranks - 0.5) / len(ranks)  # Avoid 0 and 1
                
                # Step 3: Transform to combined distribution
                target_values = stats.norm.ppf(quantiles, loc=combined_mean, scale=combined_std)
                
                # Step 4: Apply geometric interpolation (repair_level controls blend)
                repaired_values = (1 - repair_level) * original_values + repair_level * target_values
                
                # Update repaired data
                if isinstance(feature, str):
                    X_repaired.loc[mask, feature] = repaired_values
                else:
                    X_repaired.iloc[mask, feature] = repaired_values
        
        # Calculate disparate impact after repair
        if repair_level > 0.0:
            repaired_feature_data = X_repaired.iloc[:, X_repaired.columns.get_loc(feature_key) if isinstance(feature, str) else feature].values
            repaired_group_means = []
            for group in unique_groups:
                mask = protected_col == group
                repaired_group_means.append(np.mean(repaired_feature_data[mask]))
            
            di_after[feature_key] = min(repaired_group_means) / max(repaired_group_means) if max(repaired_group_means) != 0 else 1.0
        else:
            di_after[feature_key] = di_before[feature_key]
    
    # Estimate balanced error rate improvement
    # BER improvement approximated by DI improvement
    avg_di_before = np.mean(list(di_before.values())) if di_before else 1.0
    avg_di_after = np.mean(list(di_after.values())) if di_after else 1.0
    ber_improvement = avg_di_after - avg_di_before
    
    return {
        'repaired_data': X_repaired,
        'repair_level': repair_level,
        'features_repaired': features_to_repair,
        'group_statistics': group_stats,
        'disparate_impact_before': di_before,
        'disparate_impact_after': di_after,
        'balanced_error_rate_improvement': ber_improvement,
        'protected_groups': list(unique_groups),
        'n_samples': len(X_df),
        'n_features_repaired': len([f for f in features_to_repair if f in di_before])
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Create synthetic dataset with disparate impact
    n_samples = 1000
    
    # Protected attribute (0: group A, 1: group B)
    protected = np.random.binomial(1, 0.3, n_samples)
    
    # Features with disparate impact
    # Group A (protected=0) has systematically lower values
    feature1 = np.where(protected == 0, 
                        np.random.normal(2, 1, n_samples),  # Lower mean for group A
                        np.random.normal(4, 1, n_samples))  # Higher mean for group B
    
    feature2 = np.where(protected == 0,
                        np.random.normal(1, 0.8, n_samples),
                        np.random.normal(3, 0.8, n_samples))
    
    # Create DataFrame
    data = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'protected_attr': protected
    })
    
    print("Disparate Impact Removal Example")
    print("=" * 40)
    
    # Apply disparate impact removal with different repair levels
    repair_levels = [0.0, 0.5, 1.0]
    
    for repair_level in repair_levels:
        print(f"\nRepair Level: {repair_level}")
        print("-" * 20)
        
        result = disparate_impact_removal(
            X=data,
            protected_attribute='protected_attr',
            repair_level=repair_level,
            features_to_repair=['feature1', 'feature2']
        )
        
        print(f"Features repaired: {result['n_features_repaired']}")
        print(f"Protected groups: {result['protected_groups']}")
        
        print("\nDisparate Impact Ratios:")
        for feature in ['feature1', 'feature2']:
            if feature in result['disparate_impact_before']:
                di_before = result['disparate_impact_before'][feature]
                di_after = result['disparate_impact_after'][feature]
                print(f"  {feature}: {di_before:.3f} -> {di_after:.3f}")
        
        print(f"BER Improvement: {result['balanced_error_rate_improvement']:.3f}")
        
        # Show group means before and after
        if repair_level in [0.0, 1.0]:  # Show details for extreme cases
            print(f"\nGroup Statistics (repair_level={repair_level}):")
            for feature in ['feature1', 'feature2']:
                if feature in result['group_statistics']:
                    print(f"  {feature}:")
                    for group in result['group_statistics'][feature]:
                        stats = result['group_statistics'][feature][group]
                        print(f"    {group}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, n={stats['size']}")