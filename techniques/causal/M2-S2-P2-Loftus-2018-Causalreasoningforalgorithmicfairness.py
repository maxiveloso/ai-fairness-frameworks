import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

def causal_reasoning_algorithmic_fairness(
    data: pd.DataFrame,
    protected_attribute: str,
    outcome: str,
    mediators: Optional[List[str]] = None,
    confounders: Optional[List[str]] = None,
    treatment_values: Optional[List] = None,
    outcome_type: str = 'continuous',
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Perform causal reasoning for algorithmic fairness analysis.
    
    This function implements causal fairness assessment by analyzing direct and indirect
    effects of protected attributes on outcomes through specified causal pathways.
    It moves beyond observational fairness metrics by using causal inference principles
    to decompose bias into different causal pathways.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing all variables
    protected_attribute : str
        Name of the protected attribute (e.g., race, gender)
    outcome : str
        Name of the outcome variable
    mediators : List[str], optional
        List of mediator variables that lie on causal path from protected attribute to outcome
    confounders : List[str], optional
        List of confounding variables that affect both protected attribute and outcome
    treatment_values : List, optional
        Values of protected attribute to compare (default: all unique values)
    outcome_type : str, default='continuous'
        Type of outcome variable ('continuous' or 'binary')
    bootstrap_samples : int, default=1000
        Number of bootstrap samples for confidence intervals
    confidence_level : float, default=0.95
        Confidence level for intervals
        
    Returns
    -------
    Dict[str, Union[float, np.ndarray, Dict]]
        Dictionary containing:
        - 'total_effect': Total causal effect of protected attribute on outcome
        - 'direct_effect': Direct effect (not through mediators)
        - 'indirect_effect': Indirect effect through mediators
        - 'natural_direct_effect': Natural direct effect
        - 'natural_indirect_effect': Natural indirect effect
        - 'controlled_direct_effect': Controlled direct effect
        - 'proportion_mediated': Proportion of effect mediated
        - 'confidence_intervals': Bootstrap confidence intervals
        - 'fairness_metrics': Causal fairness assessment
        - 'causal_graph_info': Information about specified causal relationships
    """
    
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    
    if protected_attribute not in data.columns:
        raise ValueError(f"Protected attribute '{protected_attribute}' not found in data")
    
    if outcome not in data.columns:
        raise ValueError(f"Outcome '{outcome}' not found in data")
    
    if outcome_type not in ['continuous', 'binary']:
        raise ValueError("outcome_type must be 'continuous' or 'binary'")
    
    if mediators is None:
        mediators = []
    
    if confounders is None:
        confounders = []
    
    # Check if mediators and confounders exist in data
    for var in mediators + confounders:
        if var not in data.columns:
            raise ValueError(f"Variable '{var}' not found in data")
    
    # Remove missing values
    all_vars = [protected_attribute, outcome] + mediators + confounders
    data_clean = data[all_vars].dropna()
    
    if len(data_clean) == 0:
        raise ValueError("No complete cases available after removing missing values")
    
    # Set treatment values if not provided
    if treatment_values is None:
        treatment_values = sorted(data_clean[protected_attribute].unique())
    
    if len(treatment_values) < 2:
        raise ValueError("Need at least 2 treatment values for comparison")
    
    # Standardize continuous variables for better numerical stability
    scaler = StandardScaler()
    continuous_vars = []
    for var in mediators + confounders:
        if data_clean[var].dtype in ['float64', 'int64'] and len(data_clean[var].unique()) > 10:
            continuous_vars.append(var)
    
    if continuous_vars:
        data_clean[continuous_vars] = scaler.fit_transform(data_clean[continuous_vars])
    
    # Main causal effect estimation
    results = {}
    
    # 1. Total Effect: E[Y|do(A=a1)] - E[Y|do(A=a0)]
    total_effect = _estimate_total_effect(
        data_clean, protected_attribute, outcome, confounders, 
        treatment_values, outcome_type
    )
    results['total_effect'] = total_effect
    
    # 2. Direct and Indirect Effects (if mediators specified)
    if mediators:
        direct_effect, indirect_effect = _estimate_direct_indirect_effects(
            data_clean, protected_attribute, outcome, mediators, confounders,
            treatment_values, outcome_type
        )
        results['direct_effect'] = direct_effect
        results['indirect_effect'] = indirect_effect
        
        # Natural Direct and Indirect Effects
        nde, nie = _estimate_natural_effects(
            data_clean, protected_attribute, outcome, mediators, confounders,
            treatment_values, outcome_type
        )
        results['natural_direct_effect'] = nde
        results['natural_indirect_effect'] = nie
        
        # Controlled Direct Effect
        cde = _estimate_controlled_direct_effect(
            data_clean, protected_attribute, outcome, mediators, confounders,
            treatment_values, outcome_type
        )
        results['controlled_direct_effect'] = cde
        
        # Proportion mediated
        if abs(total_effect) > 1e-10:
            results['proportion_mediated'] = indirect_effect / total_effect
        else:
            results['proportion_mediated'] = 0.0
    else:
        results['direct_effect'] = total_effect
        results['indirect_effect'] = 0.0
        results['natural_direct_effect'] = total_effect
        results['natural_indirect_effect'] = 0.0
        results['controlled_direct_effect'] = total_effect
        results['proportion_mediated'] = 0.0
    
    # 3. Bootstrap confidence intervals
    if bootstrap_samples > 0:
        ci_results = _bootstrap_confidence_intervals(
            data_clean, protected_attribute, outcome, mediators, confounders,
            treatment_values, outcome_type, bootstrap_samples, confidence_level
        )
        results['confidence_intervals'] = ci_results
    
    # 4. Causal fairness metrics
    fairness_metrics = _compute_causal_fairness_metrics(
        data_clean, protected_attribute, outcome, mediators, confounders,
        treatment_values, results
    )
    results['fairness_metrics'] = fairness_metrics
    
    # 5. Causal graph information
    results['causal_graph_info'] = {
        'protected_attribute': protected_attribute,
        'outcome': outcome,
        'mediators': mediators,
        'confounders': confounders,
        'treatment_values': treatment_values,
        'sample_size': len(data_clean)
    }
    
    return results

def _estimate_total_effect(data: pd.DataFrame, treatment: str, outcome: str, 
                          confounders: List[str], treatment_values: List,
                          outcome_type: str) -> float:
    """Estimate total causal effect using backdoor adjustment."""
    
    if len(confounders) == 0:
        # Simple difference in means/proportions
        y1 = data[data[treatment] == treatment_values[1]][outcome].mean()
        y0 = data[data[treatment] == treatment_values[0]][outcome].mean()
        return y1 - y0
    
    # Regression adjustment for confounders
    X = data[confounders + [treatment]]
    y = data[outcome]
    
    if outcome_type == 'continuous':
        model = LinearRegression()
    else:
        model = LogisticRegression(max_iter=1000)
    
    model.fit(X, y)
    
    # Predict under different treatment values
    X_treated = X.copy()
    X_treated[treatment] = treatment_values[1]
    
    X_control = X.copy()
    X_control[treatment] = treatment_values[0]
    
    if outcome_type == 'continuous':
        y1_pred = model.predict(X_treated).mean()
        y0_pred = model.predict(X_control).mean()
    else:
        y1_pred = model.predict_proba(X_treated)[:, 1].mean()
        y0_pred = model.predict_proba(X_control)[:, 1].mean()
    
    return y1_pred - y0_pred

def _estimate_direct_indirect_effects(data: pd.DataFrame, treatment: str, outcome: str,
                                     mediators: List[str], confounders: List[str],
                                     treatment_values: List, outcome_type: str) -> Tuple[float, float]:
    """Estimate direct and indirect effects using mediation analysis."""
    
    # Step 1: Model mediators as function of treatment and confounders
    mediator_models = {}
    for mediator in mediators:
        X_med = data[confounders + [treatment]]
        y_med = data[mediator]
        
        # Determine if mediator is continuous or binary
        if len(data[mediator].unique()) <= 2:
            med_model = LogisticRegression(max_iter=1000)
        else:
            med_model = LinearRegression()
        
        med_model.fit(X_med, y_med)
        mediator_models[mediator] = med_model
    
    # Step 2: Model outcome as function of treatment, mediators, and confounders
    X_out = data[confounders + [treatment] + mediators]
    y_out = data[outcome]
    
    if outcome_type == 'continuous':
        outcome_model = LinearRegression()
    else:
        outcome_model = LogisticRegression(max_iter=1000)
    
    outcome_model.fit(X_out, y_out)
    
    # Step 3: Compute effects using g-computation
    n_samples = len(data)
    
    # Natural Direct Effect: E[Y(a1, M(a0))] - E[Y(a0, M(a0))]
    # Natural Indirect Effect: E[Y(a1, M(a1))] - E[Y(a1, M(a0))]
    
    # Predict mediators under control condition
    X_med_control = data[confounders + [treatment]].copy()
    X_med_control[treatment] = treatment_values[0]
    
    # Predict mediators under treatment condition
    X_med_treated = data[confounders + [treatment]].copy()
    X_med_treated[treatment] = treatment_values[1]
    
    # Get predicted mediator values
    M0_pred = np.zeros((n_samples, len(mediators)))
    M1_pred = np.zeros((n_samples, len(mediators)))
    
    for i, mediator in enumerate(mediators):
        model = mediator_models[mediator]
        if hasattr(model, 'predict_proba'):
            M0_pred[:, i] = model.predict_proba(X_med_control)[:, 1]
            M1_pred[:, i] = model.predict_proba(X_med_treate