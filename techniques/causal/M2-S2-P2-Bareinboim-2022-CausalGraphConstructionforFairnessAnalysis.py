import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings

def causal_graph_construction_for_fairness_analysis(
    data: pd.DataFrame,
    treatment_var: str,
    outcome_var: str,
    protected_attr: str,
    mediator_vars: Optional[List[str]] = None,
    confounder_vars: Optional[List[str]] = None,
    causal_structure: Optional[Dict[str, List[str]]] = None,
    outcome_type: str = 'continuous',
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, Union[float, Dict, List]]:
    """
    Construct causal graphs for fairness analysis using Structural Causal Models (SCM).
    
    This function implements causal graph construction for fairness analysis based on
    Bareinboim's methodology. It decomposes fairness effects into:
    - Direct discrimination (Ctf-DE): Direct effect of protected attribute on outcome
    - Indirect effects (Ctf-IE): Effects mediated through legitimate variables
    - Spurious effects (Ctf-SE): Effects due to confounding variables
    
    The method uses Pearl's causal framework and do-calculus to identify and estimate
    causal effects while accounting for different pathways of discrimination.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing all variables
    treatment_var : str
        Name of the treatment/decision variable
    outcome_var : str
        Name of the outcome variable
    protected_attr : str
        Name of the protected attribute (e.g., race, gender)
    mediator_vars : List[str], optional
        List of mediator variable names that lie on causal paths
    confounder_vars : List[str], optional
        List of confounder variable names
    causal_structure : Dict[str, List[str]], optional
        Dictionary specifying causal relationships (parent -> children)
    outcome_type : str, default='continuous'
        Type of outcome variable ('continuous' or 'binary')
    bootstrap_samples : int, default=1000
        Number of bootstrap samples for confidence intervals
    confidence_level : float, default=0.95
        Confidence level for intervals
        
    Returns:
    --------
    Dict containing:
        - 'direct_effect': Direct effect of protected attribute (Ctf-DE)
        - 'indirect_effect': Indirect effect through mediators (Ctf-IE)
        - 'spurious_effect': Spurious effect due to confounders (Ctf-SE)
        - 'total_effect': Total effect of protected attribute
        - 'causal_graph': Adjacency matrix representation
        - 'path_coefficients': Coefficients for each causal path
        - 'fairness_metrics': Various fairness measures
        - 'confidence_intervals': Bootstrap confidence intervals
        - 'p_values': Statistical significance tests
    """
    
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    
    required_vars = [treatment_var, outcome_var, protected_attr]
    missing_vars = [var for var in required_vars if var not in data.columns]
    if missing_vars:
        raise ValueError(f"Missing variables in data: {missing_vars}")
    
    if outcome_type not in ['continuous', 'binary']:
        raise ValueError("outcome_type must be 'continuous' or 'binary'")
    
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    
    # Initialize variables
    if mediator_vars is None:
        mediator_vars = []
    if confounder_vars is None:
        confounder_vars = []
    
    # Clean data
    all_vars = [treatment_var, outcome_var, protected_attr] + mediator_vars + confounder_vars
    clean_data = data[all_vars].dropna()
    
    if len(clean_data) == 0:
        raise ValueError("No complete cases after removing missing values")
    
    # Encode categorical variables
    encoded_data = clean_data.copy()
    encoders = {}
    
    for var in all_vars:
        if encoded_data[var].dtype == 'object' or encoded_data[var].dtype.name == 'category':
            encoders[var] = LabelEncoder()
            encoded_data[var] = encoders[var].fit_transform(encoded_data[var])
    
    # Construct default causal structure if not provided
    if causal_structure is None:
        causal_structure = _construct_default_causal_structure(
            protected_attr, treatment_var, outcome_var, mediator_vars, confounder_vars
        )
    
    # Build adjacency matrix for causal graph
    all_nodes = list(set([protected_attr, treatment_var, outcome_var] + mediator_vars + confounder_vars))
    adjacency_matrix = _build_adjacency_matrix(causal_structure, all_nodes)
    
    # Estimate causal effects using structural equations
    effects = _estimate_causal_effects(
        encoded_data, protected_attr, treatment_var, outcome_var,
        mediator_vars, confounder_vars, outcome_type
    )
    
    # Decompose fairness effects
    fairness_decomposition = _decompose_fairness_effects(
        encoded_data, protected_attr, treatment_var, outcome_var,
        mediator_vars, confounder_vars, causal_structure, outcome_type
    )
    
    # Calculate path coefficients
    path_coefficients = _calculate_path_coefficients(
        encoded_data, causal_structure, outcome_type
    )
    
    # Compute fairness metrics
    fairness_metrics = _compute_fairness_metrics(
        encoded_data, protected_attr, treatment_var, outcome_var, outcome_type
    )
    
    # Bootstrap confidence intervals
    confidence_intervals = _bootstrap_confidence_intervals(
        encoded_data, protected_attr, treatment_var, outcome_var,
        mediator_vars, confounder_vars, causal_structure, outcome_type,
        bootstrap_samples, confidence_level
    )
    
    # Statistical significance tests
    p_values = _compute_significance_tests(effects, encoded_data)
    
    return {
        'direct_effect': fairness_decomposition['direct_effect'],
        'indirect_effect': fairness_decomposition['indirect_effect'],
        'spurious_effect': fairness_decomposition['spurious_effect'],
        'total_effect': effects['total_effect'],
        'causal_graph': adjacency_matrix,
        'path_coefficients': path_coefficients,
        'fairness_metrics': fairness_metrics,
        'confidence_intervals': confidence_intervals,
        'p_values': p_values,
        'sample_size': len(encoded_data),
        'causal_structure': causal_structure
    }

def _construct_default_causal_structure(protected_attr: str, treatment_var: str, 
                                      outcome_var: str, mediator_vars: List[str], 
                                      confounder_vars: List[str]) -> Dict[str, List[str]]:
    """Construct default causal structure based on fairness assumptions."""
    structure = {}
    
    # Protected attribute affects treatment and outcome
    structure[protected_attr] = [treatment_var, outcome_var] + mediator_vars
    
    # Treatment affects outcome
    structure[treatment_var] = [outcome_var]
    
    # Confounders affect treatment and outcome
    for confounder in confounder_vars:
        structure[confounder] = [treatment_var, outcome_var] + mediator_vars
    
    # Mediators affect outcome
    for mediator in mediator_vars:
        structure[mediator] = [outcome_var]
    
    return structure

def _build_adjacency_matrix(causal_structure: Dict[str, List[str]], 
                          all_nodes: List[str]) -> np.ndarray:
    """Build adjacency matrix from causal structure."""
    n_nodes = len(all_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    adjacency = np.zeros((n_nodes, n_nodes))
    
    for parent, children in causal_structure.items():
        if parent in node_to_idx:
            parent_idx = node_to_idx[parent]
            for child in children:
                if child in node_to_idx:
                    child_idx = node_to_idx[child]
                    adjacency[parent_idx, child_idx] = 1
    
    return adjacency

def _estimate_causal_effects(data: pd.DataFrame, protected_attr: str, treatment_var: str,
                           outcome_var: str, mediator_vars: List[str], 
                           confounder_vars: List[str], outcome_type: str) -> Dict[str, float]:
    """Estimate causal effects using regression-based methods."""
    
    # Total effect (bivariate)
    if outcome_type == 'continuous':
        model_total = LinearRegression()
    else:
        model_total = LogisticRegression(random_state=42)
    
    X_total = data[[protected_attr]].values
    y = data[outcome_var].values
    
    model_total.fit(X_total, y)
    total_effect = model_total.coef_[0]
    
    # Controlled direct effect (adjusting for mediators and confounders)
    control_vars = mediator_vars + confounder_vars + [treatment_var]
    if control_vars:
        X_controlled = data[[protected_attr] + control_vars].values
        
        if outcome_type == 'continuous':
            model_controlled = LinearRegression()
        else:
            model_controlled = LogisticRegression(random_state=42)
        
        model_controlled.fit(X_controlled, y)
        controlled_direct_effect = model_controlled.coef_[0]
    else:
        controlled_direct_effect = total_effect
    
    return {
        'total_effect': total_effect,
        'controlled_direct_effect': controlled_direct_effect
    }

def _decompose_fairness_effects(data: pd.DataFrame, protected_attr: str, treatment_var: str,
                              outcome_var: str, mediator_vars: List[str], 
                              confounder_vars: List[str], causal_structure: Dict[str, List[str]],
                              outcome_type: str) -> Dict[str, float]:
    """Decompose fairness effects into direct, indirect, and spurious components."""
    
    # Direct effect (Ctf-DE): Effect not mediated by any variables
    control_vars = mediator_vars + confounder_vars + [treatment_var]
    if control_vars:
        X_direct = data[[protected_attr] + control_vars].values
    else:
        X_direct = data[[protected_attr]].values
    
    y = data[outcome_var].values
    
    if outcome_type == 'continuous':
        model_direct = LinearRegression()
    else:
        model_direct = LogisticRegression(random_state=42)
    
    model_direct.fit(X_direct, y)
    direct_effect = model_direct.coef_[0]
    
    # Total effect
    X_total = data[[protected_attr]].values
    if outcome_type == 'continuous':
        model_total = LinearRegression()
    else:
        model_total = LogisticRegression(random_state=42)
    
    model_total.fit(X_total, y)
    total_effect = model_total.coef_[0]
    
    # Spurious