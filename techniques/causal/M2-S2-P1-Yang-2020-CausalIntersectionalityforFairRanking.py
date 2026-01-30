import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings

def causal_intersectionality_fair_ranking(
    data: pd.DataFrame,
    outcome_col: str,
    protected_attributes: List[str],
    covariates: Optional[List[str]] = None,
    reference_group: Optional[Dict[str, Union[str, int]]] = None,
    include_interactions: bool = True,
    alpha: float = 0.05
) -> Dict:
    """
    Implement Causal Intersectionality for Fair Ranking using structural causal models.
    
    This method creates fair rankings by computing counterfactual outcomes that remove
    the causal effects of protected attributes while preserving legitimate differences.
    The approach explicitly models intersectional effects between demographic groups.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset containing outcomes, protected attributes, and covariates
    outcome_col : str
        Name of the outcome variable column
    protected_attributes : List[str]
        List of protected attribute column names (e.g., race, gender)
    covariates : Optional[List[str]]
        List of legitimate covariate column names that should influence ranking
    reference_group : Optional[Dict[str, Union[str, int]]]
        Reference group values for each protected attribute. If None, uses most frequent values
    include_interactions : bool
        Whether to include interaction terms between protected attributes
    alpha : float
        Significance level for statistical tests
        
    Returns:
    --------
    Dict containing:
        - 'original_ranking': Original ranking based on observed outcomes
        - 'fair_ranking': Fair ranking based on counterfactual outcomes
        - 'counterfactual_outcomes': Counterfactual outcome values
        - 'causal_effects': Estimated causal effects of protected attributes
        - 'model_summary': Statistical summary of the causal model
        - 'fairness_metrics': Comparison metrics between original and fair rankings
    """
    
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    
    if outcome_col not in data.columns:
        raise ValueError(f"outcome_col '{outcome_col}' not found in data")
    
    for attr in protected_attributes:
        if attr not in data.columns:
            raise ValueError(f"protected attribute '{attr}' not found in data")
    
    if covariates:
        for cov in covariates:
            if cov not in data.columns:
                raise ValueError(f"covariate '{cov}' not found in data")
    
    if data[outcome_col].isnull().any():
        raise ValueError("outcome_col contains missing values")
    
    # Create working copy of data
    df = data.copy()
    n_samples = len(df)
    
    # Encode categorical variables
    encoders = {}
    for attr in protected_attributes:
        if df[attr].dtype == 'object':
            encoders[attr] = LabelEncoder()
            df[attr] = encoders[attr].fit_transform(df[attr])
    
    # Determine reference group (most frequent category for each attribute)
    if reference_group is None:
        reference_group = {}
        for attr in protected_attributes:
            reference_group[attr] = df[attr].mode()[0]
    
    # Step 1: Build feature matrix for structural causal model
    feature_cols = []
    X_features = []
    
    # Add protected attributes
    for attr in protected_attributes:
        feature_cols.append(attr)
        X_features.append(df[attr].values)
    
    # Add covariates if provided
    if covariates:
        for cov in covariates:
            feature_cols.append(cov)
            X_features.append(df[cov].values)
    
    # Add interaction terms between protected attributes if requested
    interaction_terms = []
    if include_interactions and len(protected_attributes) > 1:
        for i, attr1 in enumerate(protected_attributes):
            for j, attr2 in enumerate(protected_attributes[i+1:], i+1):
                interaction_col = f"{attr1}_x_{attr2}"
                interaction_terms.append(interaction_col)
                feature_cols.append(interaction_col)
                X_features.append(df[attr1].values * df[attr2].values)
    
    # Construct feature matrix
    X = np.column_stack(X_features) if X_features else np.empty((n_samples, 0))
    y = df[outcome_col].values
    
    # Step 2: Estimate structural causal model parameters
    if X.shape[1] > 0:
        # Fit linear structural causal model: Y = β₀ + Σβᵢ Xᵢ + ε
        causal_model = LinearRegression()
        causal_model.fit(X, y)
        
        # Extract coefficients and statistics
        coefficients = causal_model.coef_
        intercept = causal_model.intercept_
        predictions = causal_model.predict(X)
        residuals = y - predictions
        
        # Calculate standard errors and p-values
        mse = np.mean(residuals ** 2)
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
        
        try:
            # Compute covariance matrix for coefficient standard errors
            XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            var_coef = mse * np.diag(XtX_inv)
            se_coef = np.sqrt(var_coef)
            
            # t-statistics and p-values
            t_stats = np.concatenate([[intercept], coefficients]) / se_coef
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n_samples - X.shape[1] - 1))
            
        except np.linalg.LinAlgError:
            warnings.warn("Could not compute standard errors due to singular matrix")
            se_coef = np.full(X.shape[1] + 1, np.nan)
            t_stats = np.full(X.shape[1] + 1, np.nan)
            p_values = np.full(X.shape[1] + 1, np.nan)
        
        # Model fit statistics
        r_squared = 1 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2)
        adj_r_squared = 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - X.shape[1] - 1)
        
    else:
        # No features case - use mean as prediction
        intercept = np.mean(y)
        coefficients = np.array([])
        predictions = np.full(n_samples, intercept)
        residuals = y - predictions
        r_squared = 0.0
        adj_r_squared = 0.0
        se_coef = np.array([np.std(y) / np.sqrt(n_samples)])
        t_stats = np.array([intercept / se_coef[0]])
        p_values = np.array([2 * (1 - stats.t.cdf(np.abs(t_stats[0]), df=n_samples - 1))])
    
    # Step 3: Compute counterfactual outcomes
    # Transform each observation to reference group values for protected attributes
    X_counterfactual = X.copy()
    
    for i, attr in enumerate(protected_attributes):
        ref_value = reference_group[attr]
        X_counterfactual[:, i] = ref_value
        
        # Update interaction terms involving this attribute
        if include_interactions:
            for j, interaction in enumerate(interaction_terms):
                interaction_idx = len(protected_attributes) + (len(covariates) if covariates else 0) + j
                if attr in interaction:
                    # Recompute interaction with reference value
                    attr_indices = [k for k, a in enumerate(protected_attributes) if a in interaction]
                    if len(attr_indices) == 2:
                        val1 = ref_value if attr_indices[0] == i else X_counterfactual[:, attr_indices[0]]
                        val2 = ref_value if attr_indices[1] == i else X_counterfactual[:, attr_indices[1]]
                        X_counterfactual[:, interaction_idx] = val1 * val2
    
    # Generate counterfactual predictions
    if X.shape[1] > 0:
        counterfactual_outcomes = causal_model.predict(X_counterfactual)
    else:
        counterfactual_outcomes = predictions.copy()
    
    # Step 4: Create rankings
    # Original ranking (higher outcome scores ranked first)
    original_ranking = np.argsort(-y)  # Descending order
    
    # Fair ranking based on counterfactual outcomes
    fair_ranking = np.argsort(-counterfactual_outcomes)  # Descending order
    
    # Calculate causal effects (difference between observed and counterfactual)
    individual_causal_effects = y - counterfactual_outcomes
    
    # Aggregate causal effects by protected attribute combinations
    causal_effects_summary = {}
    for attr in protected_attributes:
        attr_effects = {}
        unique_values = np.unique(df[attr])
        for val in unique_values:
            mask = df[attr] == val
            if np.sum(mask) > 0:
                attr_effects[val] = {
                    'mean_effect': np.mean(individual_causal_effects[mask]),
                    'std_effect': np.std(individual_causal_effects[mask]),
                    'count': np.sum(mask)
                }
        causal_effects_summary[attr] = attr_effects
    
    # Calculate fairness metrics
    def rank_correlation(rank1, rank2):
        """Calculate Spearman rank correlation"""
        return stats.spearmanr(rank1, rank2)[0]
    
    def positional_differences(rank1, rank2):
        """Calculate mean absolute positional difference"""
        pos1 = np.empty(len(rank1))
        pos2 = np.empty(len(rank2))
        pos1[rank1] = np.arange(len(rank1))
        pos2[rank2] = np.arange(len(rank2))
        return np.mean(np.abs(pos1 - pos2))
    
    fairness_metrics = {
        'rank_correlation': rank_correlation(original_ranking, fair_ranking),
        'mean_positional_difference': positional_differences(original_ranking, fair_ranking),
        'outcome_correlation': np.corrcoef(y, counterfactual_outcomes)[0, 1],
        'mean_causal_effect': np.mean(individual_causal_effects),
        'std_causal_effect': np.std(individual_causal_effects)
    }
    
    # Model summary
    model_summary = {
        'n_samples': n_samples,
        'n_features': X.shape[1],
        'feature_names': feature_cols,
        'coefficients': coefficients.tolist() if len(coefficients) > 0 else [],
        'intercept': intercept,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'mse': float(mse) if X.shape[1] > 0 else float(np.var(y)),
        'coefficient_pvalues': p_values[1:].tolist() if len(p_values) > 1 else [],
        'significant