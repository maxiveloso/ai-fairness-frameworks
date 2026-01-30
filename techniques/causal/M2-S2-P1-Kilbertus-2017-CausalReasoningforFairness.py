import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

def causal_reasoning_for_fairness(
    data: pd.DataFrame,
    protected_attr: str,
    outcome: str,
    features: List[str],
    proxy_vars: Optional[List[str]] = None,
    predictor: Optional[str] = None,
    causal_graph: Optional[Dict[str, List[str]]] = None,
    alpha: float = 0.05,
    standardize: bool = True
) -> Dict[str, Union[float, Dict, List]]:
    """
    Implement causal reasoning framework for fairness analysis using causal graphs.
    
    This function analyzes fairness through causal lens by decomposing bias into
    direct, indirect, and spurious components using causal graph structure.
    Based on Kilbertus et al. (2017) framework.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset containing all variables
    protected_attr : str
        Name of protected attribute (A in causal graph)
    outcome : str
        Name of outcome variable (Y in causal graph)
    features : List[str]
        List of feature variable names (X in causal graph)
    proxy_vars : Optional[List[str]]
        List of proxy variable names (P in causal graph)
    predictor : Optional[str]
        Name of predictor variable (R in causal graph)
    causal_graph : Optional[Dict[str, List[str]]]
        Causal graph structure as adjacency list
    alpha : float, default=0.05
        Significance level for statistical tests
    standardize : bool, default=True
        Whether to standardize continuous variables
        
    Returns:
    --------
    Dict containing:
        - direct_effect: Direct causal effect of protected attribute
        - indirect_effect: Indirect effect through mediators
        - spurious_effect: Spurious associations
        - total_effect: Total effect decomposition
        - fairness_metrics: Various fairness measures
        - causal_explanation: Quantified disparities
        - statistical_tests: Significance tests for effects
    """
    
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    
    required_cols = [protected_attr, outcome] + features
    if proxy_vars:
        required_cols.extend(proxy_vars)
    if predictor:
        required_cols.append(predictor)
        
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")
    
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    # Prepare data
    df = data[required_cols].copy().dropna()
    
    if len(df) == 0:
        raise ValueError("No valid observations after removing missing values")
    
    # Standardize continuous variables if requested
    if standardize:
        scaler = StandardScaler()
        continuous_vars = df.select_dtypes(include=[np.number]).columns.tolist()
        if continuous_vars:
            df[continuous_vars] = scaler.fit_transform(df[continuous_vars])
    
    # Initialize results dictionary
    results = {
        'direct_effect': {},
        'indirect_effect': {},
        'spurious_effect': {},
        'total_effect': {},
        'fairness_metrics': {},
        'causal_explanation': {},
        'statistical_tests': {}
    }
    
    # Extract variables
    A = df[protected_attr].values  # Protected attribute
    Y = df[outcome].values         # Outcome
    X = df[features].values        # Features
    
    P = df[proxy_vars].values if proxy_vars else None  # Proxy variables
    R = df[predictor].values if predictor else None    # Predictor
    
    # Determine if outcome is binary or continuous
    is_binary_outcome = len(np.unique(Y)) == 2
    
    # 1. Calculate Total Effect (A -> Y)
    if is_binary_outcome:
        model_total = LogisticRegression(random_state=42)
        model_total.fit(A.reshape(-1, 1), Y)
        total_coef = model_total.coef_[0][0]
        total_effect = np.exp(total_coef)  # Odds ratio
    else:
        model_total = LinearRegression()
        model_total.fit(A.reshape(-1, 1), Y)
        total_coef = model_total.coef_[0]
        total_effect = total_coef
    
    results['total_effect'] = {
        'coefficient': total_coef,
        'effect_size': total_effect,
        'model_score': model_total.score(A.reshape(-1, 1), Y)
    }
    
    # 2. Calculate Direct Effect (A -> Y, controlling for mediators)
    # Control for features and proxy variables
    control_vars = X
    if P is not None:
        control_vars = np.column_stack([X, P])
    
    combined_vars = np.column_stack([A.reshape(-1, 1), control_vars])
    
    if is_binary_outcome:
        model_direct = LogisticRegression(random_state=42)
        model_direct.fit(combined_vars, Y)
        direct_coef = model_direct.coef_[0][0]
        direct_effect = np.exp(direct_coef)
    else:
        model_direct = LinearRegression()
        model_direct.fit(combined_vars, Y)
        direct_coef = model_direct.coef_[0]
        direct_effect = direct_coef
    
    results['direct_effect'] = {
        'coefficient': direct_coef,
        'effect_size': direct_effect,
        'model_score': model_direct.score(combined_vars, Y)
    }
    
    # 3. Calculate Indirect Effect through mediators
    indirect_effect = total_effect - direct_effect
    
    # If proxy variables exist, calculate mediation effects
    if P is not None:
        mediation_effects = []
        for i, proxy_var in enumerate(proxy_vars):
            P_i = P[:, i]
            
            # A -> P_i (first stage)
            model_ap = LinearRegression()
            model_ap.fit(A.reshape(-1, 1), P_i)
            a_path = model_ap.coef_[0]
            
            # P_i -> Y controlling for A (second stage)
            combined_ap = np.column_stack([A.reshape(-1, 1), P_i.reshape(-1, 1)])
            if is_binary_outcome:
                model_py = LogisticRegression(random_state=42)
                model_py.fit(combined_ap, Y)
                b_path = model_py.coef_[0][1]
            else:
                model_py = LinearRegression()
                model_py.fit(combined_ap, Y)
                b_path = model_py.coef_[1]
            
            # Mediation effect = a_path * b_path
            mediation_effect = a_path * b_path
            mediation_effects.append({
                'proxy_var': proxy_var,
                'a_path': a_path,
                'b_path': b_path,
                'mediation_effect': mediation_effect
            })
        
        results['indirect_effect'] = {
            'total_indirect': indirect_effect,
            'mediation_effects': mediation_effects
        }
    else:
        results['indirect_effect'] = {
            'total_indirect': indirect_effect,
            'mediation_effects': []
        }
    
    # 4. Calculate Spurious Effect (confounding)
    # Estimate spurious associations through backdoor paths
    if len(features) > 0:
        # Effect of A on Y through confounders (features)
        spurious_effects = []
        for i, feature in enumerate(features):
            X_i = X[:, i]
            
            # A -> X_i
            model_ax = LinearRegression()
            model_ax.fit(A.reshape(-1, 1), X_i)
            ax_coef = model_ax.coef_[0]
            
            # X_i -> Y
            model_xy = LinearRegression()
            model_xy.fit(X_i.reshape(-1, 1), Y)
            xy_coef = model_xy.coef_[0]
            
            spurious_effect = ax_coef * xy_coef
            spurious_effects.append({
                'feature': feature,
                'ax_path': ax_coef,
                'xy_path': xy_coef,
                'spurious_effect': spurious_effect
            })
        
        total_spurious = sum([se['spurious_effect'] for se in spurious_effects])
        results['spurious_effect'] = {
            'total_spurious': total_spurious,
            'spurious_paths': spurious_effects
        }
    else:
        results['spurious_effect'] = {
            'total_spurious': 0.0,
            'spurious_paths': []
        }
    
    # 5. Fairness Metrics
    # Calculate group-based fairness metrics
    unique_groups = np.unique(A)
    if len(unique_groups) == 2:
        group_0_mask = A == unique_groups[0]
        group_1_mask = A == unique_groups[1]
        
        # Demographic parity
        if is_binary_outcome:
            rate_0 = np.mean(Y[group_0_mask])
            rate_1 = np.mean(Y[group_1_mask])
            demographic_parity = rate_1 - rate_0
        else:
            mean_0 = np.mean(Y[group_0_mask])
            mean_1 = np.mean(Y[group_1_mask])
            demographic_parity = mean_1 - mean_0
        
        # Statistical parity test
        if is_binary_outcome:
            stat, p_value = stats.chi2_contingency(
                pd.crosstab(A, Y)
            )[:2]
        else:
            stat, p_value = stats.ttest_ind(
                Y[group_1_mask], Y[group_0_mask]
            )
        
        results['fairness_metrics'] = {
            'demographic_parity': demographic_parity,
            'statistical_test': {
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < alpha
            }
        }
    
    # 6. Causal Explanation Formula
    # Quantify disparities using causal decomposition
    causal_explanation = {
        'total_disparity': total_effect,
        'direct_discrimination': direct_effect,
        'indirect_discrimination': results['indirect_effect']['total_indirect'],
        'spurious_correlation': results['spurious_effect']['total_spurious']
    }
    
    # Calculate proportions
    if abs(total_effect) > 1e-10:
        causal_explanation['direct_proportion'] = abs(direct_effect) / abs(total_effect)
        causal_explanation['indirect_proportion'] = abs(results['indirect_effect']['total_indirect']) / abs(total_effect)
        causal_explanation['spurious_proportion'] = abs(results['spurious_effect']['total_spurious']) / abs(total_effect)
    else:
        causal_explanation['direct_proportion'] = 0.0
        causal_explanation['indirect_proportion'] = 0.