import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy import optimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

def counterfactual_fairness(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    protected_attr: Union[pd.Series, np.ndarray, str],
    causal_graph: Dict[str, List[str]],
    predictor: Optional[BaseEstimator] = None,
    n_samples: int = 1000,
    lambda_penalty: float = 1.0,
    max_iter: int = 100,
    tolerance: float = 1e-6,
    random_state: Optional[int] = None
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Implement Counterfactual Fairness as proposed by Kusner et al. (2017).
    
    Counterfactual fairness ensures that a predictor's decision for an individual
    would be the same in a counterfactual world where the individual belonged to
    a different demographic group, holding all other causally relevant factors constant.
    
    The method requires:
    1. A causal DAG specifying relationships between variables
    2. Identification of non-descendants of the protected attribute
    3. Training predictors that only use causally admissible features
    4. Measuring counterfactual differences through causal interventions
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : pd.Series or np.ndarray
        Target variable of shape (n_samples,)
    protected_attr : pd.Series, np.ndarray, or str
        Protected attribute values or column name if X is DataFrame
    causal_graph : Dict[str, List[str]]
        Causal DAG as adjacency list where keys are parents and values are children
    predictor : BaseEstimator, optional
        Sklearn-compatible predictor. If None, uses LogisticRegression
    n_samples : int, default=1000
        Number of samples for counterfactual generation
    lambda_penalty : float, default=1.0
        Penalty weight for counterfactual differences in optimization
    max_iter : int, default=100
        Maximum iterations for optimization
    tolerance : float, default=1e-6
        Convergence tolerance for optimization
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Dict[str, Union[float, np.ndarray, Dict]]
        Dictionary containing:
        - 'counterfactual_fairness_score': Overall fairness score (0=fair, higher=unfair)
        - 'individual_cf_differences': Per-individual counterfactual differences
        - 'admissible_features': Features that are causally admissible
        - 'causal_model_params': Parameters of the fitted causal model
        - 'predictor_performance': Performance metrics of the fair predictor
        - 'counterfactual_violations': Number/percentage of violations
        
    References
    ----------
    Kusner, M. J., Loftus, J., Russell, C., & Silva, R. (2017). 
    Counterfactual fairness. In Advances in Neural Information Processing Systems 
    (pp. 4066-4076).
    """
    
    # Input validation
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_array = X.values
    else:
        X_array = np.asarray(X)
        feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
    
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = np.asarray(y)
    
    if isinstance(protected_attr, str):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("If protected_attr is string, X must be DataFrame")
        A = X[protected_attr].values
        protected_idx = feature_names.index(protected_attr)
    else:
        A = np.asarray(protected_attr)
        protected_idx = None
    
    if X_array.shape[0] != len(y_array) or X_array.shape[0] != len(A):
        raise ValueError("X, y, and protected_attr must have same number of samples")
    
    if not isinstance(causal_graph, dict):
        raise ValueError("causal_graph must be a dictionary")
    
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize predictor
    if predictor is None:
        predictor = LogisticRegression(random_state=random_state)
    
    n_samples_data, n_features = X_array.shape
    
    # Step 1: Identify causally admissible features (non-descendants of protected attribute)
    admissible_features = _identify_admissible_features(
        feature_names, protected_attr, causal_graph
    )
    
    # Get indices of admissible features
    admissible_indices = [i for i, name in enumerate(feature_names) 
                         if name in admissible_features]
    
    if len(admissible_indices) == 0:
        warnings.warn("No admissible features found. Using all features except protected attribute.")
        admissible_indices = [i for i in range(n_features) if i != protected_idx]
        admissible_features = [feature_names[i] for i in admissible_indices]
    
    X_admissible = X_array[:, admissible_indices]
    
    # Step 2: Fit causal model for latent variables and structural equations
    causal_model_params = _fit_causal_model(
        X_array, A, feature_names, causal_graph, random_state
    )
    
    # Step 3: Train counterfactually fair predictor
    # The predictor should only use admissible features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_admissible)
    
    # Optimize predictor with counterfactual fairness penalty
    fair_predictor = _train_fair_predictor(
        X_scaled, y_array, A, predictor, lambda_penalty, 
        max_iter, tolerance, random_state
    )
    
    # Step 4: Generate counterfactual samples and measure fairness
    cf_results = _measure_counterfactual_fairness(
        X_scaled, A, fair_predictor, causal_model_params, 
        n_samples, random_state
    )
    
    # Step 5: Evaluate predictor performance
    y_pred = fair_predictor.predict(X_scaled)
    y_pred_proba = fair_predictor.predict_proba(X_scaled)[:, 1] if hasattr(fair_predictor, 'predict_proba') else y_pred
    
    performance = {
        'accuracy': np.mean(y_pred == y_array),
        'mean_prediction': np.mean(y_pred_proba),
        'prediction_std': np.std(y_pred_proba)
    }
    
    # Calculate overall fairness score
    individual_differences = cf_results['individual_differences']
    fairness_score = np.mean(np.abs(individual_differences))
    
    # Count violations (differences above threshold)
    threshold = 0.1  # 10% difference threshold
    violations = np.sum(np.abs(individual_differences) > threshold)
    violation_rate = violations / len(individual_differences)
    
    return {
        'counterfactual_fairness_score': fairness_score,
        'individual_cf_differences': individual_differences,
        'admissible_features': admissible_features,
        'causal_model_params': causal_model_params,
        'predictor_performance': performance,
        'counterfactual_violations': {
            'count': int(violations),
            'rate': violation_rate,
            'threshold': threshold
        },
        'n_admissible_features': len(admissible_features),
        'total_features': n_features
    }


def _identify_admissible_features(
    feature_names: List[str], 
    protected_attr: Union[str, int], 
    causal_graph: Dict[str, List[str]]
) -> List[str]:
    """
    Identify features that are not descendants of the protected attribute.
    
    In counterfactual fairness, we can only use features that are not
    causally downstream of the protected attribute, as these would change
    in counterfactual scenarios.
    """
    if isinstance(protected_attr, str):
        protected_name = protected_attr
    else:
        protected_name = f'feature_{protected_attr}'
    
    # Find all descendants of protected attribute using BFS
    descendants = set()
    queue = [protected_name]
    
    while queue:
        current = queue.pop(0)
        if current in causal_graph:
            for child in causal_graph[current]:
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)
    
    # Admissible features are those not descended from protected attribute
    admissible = [name for name in feature_names 
                 if name != protected_name and name not in descendants]
    
    return admissible


def _fit_causal_model(
    X: np.ndarray, 
    A: np.ndarray, 
    feature_names: List[str], 
    causal_graph: Dict[str, List[str]],
    random_state: Optional[int]
) -> Dict[str, np.ndarray]:
    """
    Fit structural equation model for causal relationships.
    
    This simplified implementation assumes linear relationships and
    Gaussian noise for the latent variables U.
    """
    n_samples, n_features = X.shape
    
    # Initialize latent variables (simplified as Gaussian)
    if random_state is not None:
        np.random.seed(random_state)
    
    U = np.random.normal(0, 1, (n_samples, n_features))
    
    # Fit linear coefficients for each variable given its parents
    coefficients = {}
    noise_vars = {}
    
    for i, feature in enumerate(feature_names):
        # Find parents in causal graph
        parents = []
        for parent, children in causal_graph.items():
            if feature in children:
                parents.append(parent)
        
        if parents:
            # Get parent indices
            parent_indices = [feature_names.index(p) for p in parents if p in feature_names]
            
            if parent_indices:
                X_parents = X[:, parent_indices]
                # Add latent variable
                X_with_latent = np.column_stack([X_parents, U[:, i]])
                
                # Fit linear regression
                coef = np.linalg.lstsq(X_with_latent, X[:, i], rcond=None)[0]
                coefficients[feature] = coef
                
                # Estimate noise variance
                y_pred = X_with_latent @ coef
                noise_vars[feature] = np.var(X[:, i] - y_pred)
            else:
                # No valid parents, use mean and variance
                coefficients[feature] = np.array([np.mean(X[:, i])])
                noise_vars[feature] = np.var(X[:, i])
        else:
            # No parents, use mean and variance
            coefficients[feature] = np.array([np.mean(X[:, i])])
            noise_vars[feature] = np.var(X