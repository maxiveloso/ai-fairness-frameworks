import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings

def intersectional_counterfactual_queries(
    data: pd.DataFrame,
    outcome_col: str,
    protected_attributes: List[str],
    covariates: List[str],
    reference_group: Dict[str, Union[str, int]],
    model_type: str = 'linear',
    causal_graph: Optional[Dict[str, List[str]]] = None,
    n_bootstrap: int = 100,
    random_state: Optional[int] = None
) -> Dict[str, Union[float, np.ndarray, pd.DataFrame]]:
    """
    Implement Intersectional Counterfactual Queries for fair ranking.
    
    This method estimates counterfactual outcomes by modeling what would happen
    if individuals from different demographic groups had belonged to a reference
    group, enabling fair comparisons across intersectional identities.
    
    The algorithm follows three steps:
    1. Estimate causal model relating protected attributes to outcomes
    2. Generate counterfactual data by transforming observations to reference group
    3. Rank using counterfactual scores for fair comparison
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset with observations
    outcome_col : str
        Name of the outcome variable column
    protected_attributes : List[str]
        List of protected attribute column names (e.g., ['race', 'gender'])
    covariates : List[str]
        List of covariate column names used in causal model
    reference_group : Dict[str, Union[str, int]]
        Reference group values for each protected attribute
    model_type : str, default='linear'
        Type of causal model ('linear' or 'forest')
    causal_graph : Optional[Dict[str, List[str]]], default=None
        Causal graph structure as adjacency list
    n_bootstrap : int, default=100
        Number of bootstrap samples for uncertainty estimation
    random_state : Optional[int], default=None
        Random seed for reproducibility
        
    Returns:
    --------
    Dict containing:
        - 'counterfactual_outcomes': Estimated counterfactual outcomes
        - 'original_outcomes': Original observed outcomes
        - 'fairness_gap': Difference between original and counterfactual outcomes
        - 'ranking_original': Original ranking based on observed outcomes
        - 'ranking_counterfactual': Fair ranking based on counterfactual outcomes
        - 'group_effects': Average treatment effects by demographic group
        - 'bootstrap_ci': Bootstrap confidence intervals for effects
        - 'model_performance': R-squared of causal model
    """
    
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    
    if outcome_col not in data.columns:
        raise ValueError(f"outcome_col '{outcome_col}' not found in data")
    
    for attr in protected_attributes:
        if attr not in data.columns:
            raise ValueError(f"Protected attribute '{attr}' not found in data")
    
    for cov in covariates:
        if cov not in data.columns:
            raise ValueError(f"Covariate '{cov}' not found in data")
    
    if not all(attr in reference_group for attr in protected_attributes):
        raise ValueError("reference_group must contain all protected attributes")
    
    if model_type not in ['linear', 'forest']:
        raise ValueError("model_type must be 'linear' or 'forest'")
    
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Prepare data
    data_clean = data.dropna(subset=[outcome_col] + protected_attributes + covariates)
    n_obs = len(data_clean)
    
    if n_obs < 10:
        raise ValueError("Insufficient data after removing missing values")
    
    # Step 1: Estimate causal model
    # Create feature matrix including protected attributes and covariates
    feature_cols = protected_attributes + covariates
    X = pd.get_dummies(data_clean[feature_cols], drop_first=False)
    y = data_clean[outcome_col].values
    
    # Fit causal model
    if model_type == 'linear':
        causal_model = LinearRegression()
    else:  # forest
        causal_model = RandomForestRegressor(
            n_estimators=100, 
            random_state=random_state
        )
    
    causal_model.fit(X, y)
    model_score = causal_model.score(X, y)
    
    # Step 2: Generate counterfactual data
    # Create counterfactual dataset where all observations have reference group values
    data_counterfactual = data_clean.copy()
    
    for attr, ref_val in reference_group.items():
        data_counterfactual[attr] = ref_val
    
    # Transform counterfactual data to same format as training data
    X_counterfactual = pd.get_dummies(
        data_counterfactual[feature_cols], 
        drop_first=False
    )
    
    # Ensure same columns as training data
    missing_cols = set(X.columns) - set(X_counterfactual.columns)
    for col in missing_cols:
        X_counterfactual[col] = 0
    
    extra_cols = set(X_counterfactual.columns) - set(X.columns)
    for col in extra_cols:
        X_counterfactual = X_counterfactual.drop(columns=[col])
    
    X_counterfactual = X_counterfactual[X.columns]  # Ensure same order
    
    # Predict counterfactual outcomes
    y_counterfactual = causal_model.predict(X_counterfactual)
    
    # Step 3: Calculate fairness metrics and rankings
    y_original = y
    fairness_gap = y_original - y_counterfactual
    
    # Rankings (higher values get better ranks)
    ranking_original = stats.rankdata(-y_original, method='ordinal')
    ranking_counterfactual = stats.rankdata(-y_counterfactual, method='ordinal')
    
    # Calculate group effects
    group_effects = {}
    for group_combo in data_clean[protected_attributes].drop_duplicates().itertuples(index=False):
        # Create mask for this demographic group
        mask = np.ones(len(data_clean), dtype=bool)
        for i, attr in enumerate(protected_attributes):
            mask &= (data_clean[attr] == group_combo[i])
        
        if mask.sum() > 0:
            group_key = tuple(group_combo)
            group_effects[group_key] = {
                'n_obs': mask.sum(),
                'mean_original': np.mean(y_original[mask]),
                'mean_counterfactual': np.mean(y_counterfactual[mask]),
                'mean_gap': np.mean(fairness_gap[mask]),
                'std_gap': np.std(fairness_gap[mask])
            }
    
    # Bootstrap confidence intervals
    bootstrap_gaps = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        boot_indices = np.random.choice(n_obs, size=n_obs, replace=True)
        X_boot = X.iloc[boot_indices]
        y_boot = y[boot_indices]
        X_cf_boot = X_counterfactual.iloc[boot_indices]
        
        # Fit model on bootstrap sample
        if model_type == 'linear':
            boot_model = LinearRegression()
        else:
            boot_model = RandomForestRegressor(
                n_estimators=100, 
                random_state=random_state
            )
        
        try:
            boot_model.fit(X_boot, y_boot)
            y_cf_boot = boot_model.predict(X_cf_boot)
            gap_boot = y_boot - y_cf_boot
            bootstrap_gaps.append(np.mean(gap_boot))
        except:
            # Handle potential fitting issues in bootstrap
            bootstrap_gaps.append(np.nan)
    
    bootstrap_gaps = np.array(bootstrap_gaps)
    bootstrap_gaps = bootstrap_gaps[~np.isnan(bootstrap_gaps)]
    
    if len(bootstrap_gaps) > 0:
        bootstrap_ci = {
            'lower': np.percentile(bootstrap_gaps, 2.5),
            'upper': np.percentile(bootstrap_gaps, 97.5),
            'mean': np.mean(bootstrap_gaps),
            'std': np.std(bootstrap_gaps)
        }
    else:
        bootstrap_ci = {
            'lower': np.nan,
            'upper': np.nan,
            'mean': np.nan,
            'std': np.nan
        }
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'original_outcome': y_original,
        'counterfactual_outcome': y_counterfactual,
        'fairness_gap': fairness_gap,
        'original_rank': ranking_original,
        'counterfactual_rank': ranking_counterfactual,
        'rank_change': ranking_original - ranking_counterfactual
    })
    
    # Add protected attributes to results
    for attr in protected_attributes:
        results_df[attr] = data_clean[attr].values
    
    return {
        'counterfactual_outcomes': y_counterfactual,
        'original_outcomes': y_original,
        'fairness_gap': fairness_gap,
        'ranking_original': ranking_original,
        'ranking_counterfactual': ranking_counterfactual,
        'group_effects': group_effects,
        'bootstrap_ci': bootstrap_ci,
        'model_performance': model_score,
        'results_dataframe': results_df,
        'mean_fairness_gap': np.mean(fairness_gap),
        'std_fairness_gap': np.std(fairness_gap),
        'max_rank_change': np.max(np.abs(ranking_original - ranking_counterfactual))
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic dataset with intersectional bias
    n_samples = 500
    
    # Protected attributes
    race = np.random.choice(['White', 'Black', 'Hispanic'], size=n_samples, p=[0.6, 0.2, 0.2])
    gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.5, 0.5])
    
    # Covariates
    education = np.random.normal(12, 3, n_samples)  # Years of education
    experience = np.random.exponential(5, n_samples)  # Years of experience
    
    # Outcome with intersectional bias
    # Base outcome depends on qualifications
    base_outcome = 2 * education + 0.5 * experience + np.random.normal(0, 2, n_samples)
    
    # Add intersectional bias
    bias = np.zeros(n_samples)
    bias += np.where(race == 'Black', -3, 0)  # Race penalty
    bias += np.where(race == 'Hispanic', -2, 0)
    bias += np.where(gender == 'Female', -1.5, 0)  # Gender penalty
    bias += np.where((race == 'Black') & (gender == 'Female'), -1, 0)  # Intersectional penalty
    
    outcome = base_outcome + bias + np.random.normal(0,