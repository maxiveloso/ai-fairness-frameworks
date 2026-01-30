import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

def sensitivity_analysis_counterfactual_fairness(
    X: np.ndarray,
    y: np.ndarray,
    protected_attr: np.ndarray,
    causal_graph: Dict[str, List[str]],
    sensitivity_params: Dict[str, float],
    predictor_features: Optional[List[int]] = None,
    lambda_penalty: float = 0.1,
    max_iter: int = 1000,
    tolerance: float = 1e-6,
    confidence_level: float = 0.95
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Perform sensitivity analysis for counterfactual fairness assessment.
    
    This function implements the Wu et al. (2019) approach for analyzing counterfactual
    fairness when some causal effects are unidentifiable. It computes bounds on
    path-specific counterfactual effects and optimizes a fair predictor using
    penalized maximum likelihood with constraints.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix including all observed variables
    y : np.ndarray of shape (n_samples,)
        Target variable (binary outcomes)
    protected_attr : np.ndarray of shape (n_samples,)
        Protected attribute values (binary)
    causal_graph : Dict[str, List[str]]
        Causal graph structure where keys are variable names and values are lists
        of their causal children. Variables should be named as 'X0', 'X1', etc.
        for features, 'A' for protected attribute, 'Y' for outcome
    sensitivity_params : Dict[str, float]
        Sensitivity parameters for bounding unidentifiable quantities.
        Keys should include 'gamma_direct', 'gamma_indirect' for path-specific bounds
    predictor_features : Optional[List[int]], default=None
        Indices of features that can be used in fair predictor (non-descendants of A).
        If None, automatically determined from causal graph
    lambda_penalty : float, default=0.1
        Penalty parameter for regularization in optimization
    max_iter : int, default=1000
        Maximum iterations for optimization
    tolerance : float, default=1e-6
        Convergence tolerance for optimization
    confidence_level : float, default=0.95
        Confidence level for bound intervals
        
    Returns
    -------
    Dict[str, Union[float, np.ndarray, Dict]]
        Dictionary containing:
        - 'counterfactual_bounds': Dict with upper and lower bounds for each path
        - 'path_specific_effects': Dict with estimated effects for identifiable paths
        - 'fair_predictor_coefs': Coefficients of the optimized fair predictor
        - 'fairness_violation_bounds': Upper and lower bounds on fairness violations
        - 'optimization_status': Status of the penalized ML optimization
        - 'sensitivity_analysis': Results varying sensitivity parameters
        
    References
    ----------
    Wu, Y., Zhang, L., & Wu, X. (2019). Counterfactual fairness: Unidentification, 
    bound and algorithm. IJCAI 2019.
    """
    
    # Input validation
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    
    if X.shape[0] != y.shape[0] or X.shape[0] != protected_attr.shape[0]:
        raise ValueError("X, y, and protected_attr must have same number of samples")
    
    if not isinstance(causal_graph, dict):
        raise TypeError("causal_graph must be a dictionary")
    
    if not isinstance(sensitivity_params, dict):
        raise TypeError("sensitivity_params must be a dictionary")
    
    required_params = ['gamma_direct', 'gamma_indirect']
    if not all(param in sensitivity_params for param in required_params):
        raise ValueError(f"sensitivity_params must contain: {required_params}")
    
    n_samples, n_features = X.shape
    
    # Standardize features for numerical stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 1: Identify causal paths from protected attribute to outcome
    paths = _identify_causal_paths(causal_graph, 'A', 'Y')
    
    # Step 2: Determine non-descendants of protected attribute for fair predictor
    if predictor_features is None:
        predictor_features = _get_non_descendants(causal_graph, 'A', n_features)
    
    # Step 3: Compute path-specific counterfactual effects
    path_effects = {}
    identifiable_effects = {}
    
    for i, path in enumerate(paths):
        # Check if path effect is identifiable
        if _is_path_identifiable(path, causal_graph):
            effect = _compute_identifiable_path_effect(
                X_scaled, y, protected_attr, path, causal_graph
            )
            identifiable_effects[f'path_{i}'] = effect
        else:
            # Use sensitivity parameters for unidentifiable effects
            if 'direct' in path or len(path) == 2:  # Direct path A -> Y
                gamma = sensitivity_params['gamma_direct']
            else:  # Indirect paths
                gamma = sensitivity_params['gamma_indirect']
            
            path_effects[f'path_{i}'] = {
                'path': path,
                'identifiable': False,
                'sensitivity_param': gamma
            }
    
    # Step 4: Compute bounds for unidentifiable counterfactual quantities
    counterfactual_bounds = _compute_counterfactual_bounds(
        X_scaled, y, protected_attr, path_effects, sensitivity_params, confidence_level
    )
    
    # Step 5: Optimize fair predictor using penalized maximum likelihood
    fair_predictor_result = _optimize_fair_predictor(
        X_scaled[:, predictor_features], y, protected_attr,
        lambda_penalty, max_iter, tolerance
    )
    
    # Step 6: Compute fairness violation bounds
    fairness_bounds = _compute_fairness_violation_bounds(
        X_scaled, y, protected_attr, counterfactual_bounds, 
        fair_predictor_result['coefficients'], predictor_features
    )
    
    # Step 7: Sensitivity analysis - vary parameters
    sensitivity_analysis = _perform_sensitivity_analysis(
        X_scaled, y, protected_attr, causal_graph, sensitivity_params,
        predictor_features, lambda_penalty
    )
    
    return {
        'counterfactual_bounds': counterfactual_bounds,
        'path_specific_effects': identifiable_effects,
        'fair_predictor_coefs': fair_predictor_result['coefficients'],
        'fairness_violation_bounds': fairness_bounds,
        'optimization_status': fair_predictor_result['status'],
        'sensitivity_analysis': sensitivity_analysis,
        'causal_paths': paths,
        'predictor_features': predictor_features,
        'log_likelihood': fair_predictor_result['log_likelihood']
    }


def _identify_causal_paths(graph: Dict[str, List[str]], source: str, target: str) -> List[List[str]]:
    """Identify all causal paths from source to target in the graph."""
    paths = []
    
    def dfs(current, path, visited):
        if current == target:
            paths.append(path.copy())
            return
        
        if current in visited:
            return
            
        visited.add(current)
        
        for child in graph.get(current, []):
            path.append(child)
            dfs(child, path, visited.copy())
            path.pop()
    
    dfs(source, [source], set())
    return paths


def _get_non_descendants(graph: Dict[str, List[str]], protected_attr: str, n_features: int) -> List[int]:
    """Get indices of features that are non-descendants of protected attribute."""
    descendants = set()
    
    def find_descendants(node):
        for child in graph.get(node, []):
            if child not in descendants:
                descendants.add(child)
                find_descendants(child)
    
    find_descendants(protected_attr)
    
    # Convert to feature indices (assuming features are named X0, X1, ...)
    non_descendant_indices = []
    for i in range(n_features):
        feature_name = f'X{i}'
        if feature_name not in descendants:
            non_descendant_indices.append(i)
    
    return non_descendant_indices


def _is_path_identifiable(path: List[str], graph: Dict[str, List[str]]) -> bool:
    """Check if a causal path effect is identifiable from observational data."""
    # Simplified identifiability check - in practice this would be more complex
    # Direct paths are often identifiable, complex indirect paths may not be
    if len(path) <= 2:
        return True
    
    # Check for confounders (simplified)
    for i in range(len(path) - 1):
        current = path[i]
        next_node = path[i + 1]
        # If there are multiple parents, might have confounding
        parents = [k for k, v in graph.items() if next_node in v]
        if len(parents) > 1:
            return False
    
    return True


def _compute_identifiable_path_effect(
    X: np.ndarray, y: np.ndarray, protected_attr: np.ndarray,
    path: List[str], graph: Dict[str, List[str]]
) -> Dict[str, float]:
    """Compute path-specific effect for identifiable paths."""
    # Use regression-based approach for path effect estimation
    
    # For direct effect A -> Y, use regression controlling for mediators
    if len(path) == 2:  # Direct path
        # Control for all mediators
        mediators = []
        for node in graph.get('A', []):
            if node != 'Y' and node.startswith('X'):
                idx = int(node[1:])
                mediators.append(idx)
        
        if mediators:
            X_control = np.column_stack([protected_attr, X[:, mediators]])
        else:
            X_control = protected_attr.reshape(-1, 1)
        
        # Fit logistic regression
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(fit_intercept=True, max_iter=1000)
        model.fit(X_control, y)
        
        # Direct effect is coefficient of protected attribute
        direct_effect = model.coef_[0][0] if X_control.shape[1] > 1 else model.coef_[0]
        
        return {
            'effect_size': direct_effect,
            'standard_error': np.sqrt(np.diag(np.linalg.inv(X_control.T @ X_control + 1e-6 * np.eye(X_control.shape[1]))))[0],
            'path': path
        }
    
    else:  # Indirect path - use mediation analysis
        # Simplified indirect effect computation
        # Product of path coefficients method
        effect_product = 1.0
        se_product = 0.0
        
        for i in range(len(path) - 1):
            # Estimate coefficient for each edge in path
            current = path[i]
            next_node = path[i + 1]
            
            if current == 'A':
                coef