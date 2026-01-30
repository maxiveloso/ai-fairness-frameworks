import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy import optimize
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings

def counterfactual_fairness(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    protected_attr: Union[np.ndarray, pd.Series, str],
    causal_graph: Dict[str, List[str]],
    feature_names: Optional[List[str]] = None,
    predictor_func: Optional[Callable] = None,
    n_latent: int = 5,
    max_iter: int = 1000,
    penalty_weight: float = 1.0,
    tolerance: float = 1e-6,
    random_state: Optional[int] = None
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Implement Counterfactual Fairness as proposed by Kusner et al. (2017).
    
    Counterfactual fairness ensures that a predictor would give the same output
    in a counterfactual world where the protected attribute had been different,
    all else being equal. This is achieved by constraining the predictor to only
    use non-descendants of the protected attribute in the causal graph.
    
    The method involves:
    1. Identifying non-descendants of protected attribute A in causal graph G
    2. Learning latent variables U that explain the data generation process
    3. Training predictor Ŷ = g_θ(U, X⊁A) using only non-descendants X⊁A
    4. Optimizing with penalty for counterfactual unfairness
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target variable
    protected_attr : array-like of shape (n_samples,) or str
        Protected attribute values or column name if X is DataFrame
    causal_graph : dict
        Causal graph as adjacency list {node: [children]}
    feature_names : list of str, optional
        Names of features in X
    predictor_func : callable, optional
        Custom predictor function, defaults to linear regression
    n_latent : int, default=5
        Number of latent variables to learn
    max_iter : int, default=1000
        Maximum iterations for optimization
    penalty_weight : float, default=1.0
        Weight for counterfactual fairness penalty
    tolerance : float, default=1e-6
        Convergence tolerance
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'counterfactual_fairness_score': float, fairness score (0 = perfectly fair)
        - 'predictor_params': array, learned predictor parameters
        - 'latent_variables': array, learned latent variables U
        - 'non_descendants': list, features identified as non-descendants of A
        - 'fairness_penalty': float, penalty term from optimization
        - 'prediction_loss': float, prediction accuracy loss
        - 'total_loss': float, combined loss (prediction + fairness penalty)
        - 'converged': bool, whether optimization converged
        - 'n_iterations': int, number of optimization iterations
    """
    
    # Input validation
    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = X.columns.tolist()
        X_array = X.values
    else:
        X_array = np.asarray(X)
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
    
    if isinstance(protected_attr, str):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("protected_attr as string requires X to be DataFrame")
        protected_values = X[protected_attr].values
        protected_name = protected_attr
    else:
        protected_values = np.asarray(protected_attr)
        protected_name = 'protected_attr'
    
    y_array = np.asarray(y)
    
    if X_array.shape[0] != len(y_array) or X_array.shape[0] != len(protected_values):
        raise ValueError("X, y, and protected_attr must have same number of samples")
    
    if not isinstance(causal_graph, dict):
        raise ValueError("causal_graph must be a dictionary")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Identify non-descendants of protected attribute in causal graph
    def find_descendants(graph: Dict[str, List[str]], node: str) -> set:
        """Find all descendants of a node in the causal graph."""
        descendants = set()
        stack = [node]
        
        while stack:
            current = stack.pop()
            if current in graph:
                for child in graph[current]:
                    if child not in descendants:
                        descendants.add(child)
                        stack.append(child)
        
        return descendants
    
    # Find descendants of protected attribute
    protected_descendants = find_descendants(causal_graph, protected_name)
    
    # Identify non-descendant features
    non_descendant_indices = []
    non_descendant_names = []
    
    for i, feature_name in enumerate(feature_names):
        if feature_name != protected_name and feature_name not in protected_descendants:
            non_descendant_indices.append(i)
            non_descendant_names.append(feature_name)
    
    if len(non_descendant_indices) == 0:
        warnings.warn("No non-descendant features found. Using all features except protected attribute.")
        non_descendant_indices = [i for i, name in enumerate(feature_names) if name != protected_name]
        non_descendant_names = [name for name in feature_names if name != protected_name]
    
    # Extract non-descendant features
    X_non_desc = X_array[:, non_descendant_indices]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_non_desc)
    y_scaled = (y_array - np.mean(y_array)) / np.std(y_array)
    
    n_samples, n_features = X_scaled.shape
    
    # Initialize latent variables and parameters
    U = np.random.normal(0, 1, (n_samples, n_latent))
    
    # Combine latent variables with non-descendant features
    X_combined = np.column_stack([U, X_scaled])
    n_combined_features = X_combined.shape[1]
    
    # Initialize predictor parameters
    theta = np.random.normal(0, 0.1, n_combined_features)
    
    def compute_counterfactual_penalty(U: np.ndarray, X_nd: np.ndarray, 
                                     theta: np.ndarray, protected: np.ndarray) -> float:
        """
        Compute penalty for counterfactual unfairness.
        
        The penalty measures the difference in predictions between actual and
        counterfactual worlds where protected attribute values are swapped.
        """
        X_combined = np.column_stack([U, X_nd])
        predictions = X_combined @ theta
        
        # Create counterfactual by swapping protected attribute values
        # Since we use non-descendants, the counterfactual difference comes
        # through the latent variables that may be influenced by the protected attribute
        unique_protected = np.unique(protected)
        if len(unique_protected) < 2:
            return 0.0
        
        penalty = 0.0
        n_comparisons = 0
        
        # Compare predictions for individuals with different protected attribute values
        # but similar non-descendant features (proxy for counterfactual comparison)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if protected[i] != protected[j]:
                    # Weight by similarity in non-descendant features
                    feature_diff = np.linalg.norm(X_nd[i] - X_nd[j])
                    if feature_diff < np.percentile(np.linalg.norm(X_nd[:, None] - X_nd[None, :], axis=2), 25):
                        pred_diff = abs(predictions[i] - predictions[j])
                        weight = np.exp(-feature_diff)  # Higher weight for more similar individuals
                        penalty += weight * pred_diff
                        n_comparisons += weight
        
        return penalty / max(n_comparisons, 1)
    
    def objective_function(params: np.ndarray) -> float:
        """
        Objective function combining prediction loss and fairness penalty.
        
        The function optimizes predictor parameters and latent variables jointly
        to minimize prediction error while ensuring counterfactual fairness.
        """
        # Split parameters
        theta_current = params[:n_combined_features]
        U_flat = params[n_combined_features:]
        U_current = U_flat.reshape(n_samples, n_latent)
        
        # Prediction loss
        X_combined = np.column_stack([U_current, X_scaled])
        predictions = X_combined @ theta_current
        prediction_loss = np.mean((predictions - y_scaled) ** 2)
        
        # Counterfactual fairness penalty
        fairness_penalty = compute_counterfactual_penalty(
            U_current, X_scaled, theta_current, protected_values
        )
        
        total_loss = prediction_loss + penalty_weight * fairness_penalty
        return total_loss
    
    # Initialize optimization parameters
    initial_params = np.concatenate([theta, U.flatten()])
    
    # Optimize using scipy
    result = optimize.minimize(
        objective_function,
        initial_params,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tolerance}
    )
    
    # Extract optimized parameters
    theta_opt = result.x[:n_combined_features]
    U_opt = result.x[n_combined_features:].reshape(n_samples, n_latent)
    
    # Compute final metrics
    X_combined_opt = np.column_stack([U_opt, X_scaled])
    final_predictions = X_combined_opt @ theta_opt
    final_prediction_loss = np.mean((final_predictions - y_scaled) ** 2)
    final_fairness_penalty = compute_counterfactual_penalty(
        U_opt, X_scaled, theta_opt, protected_values
    )
    
    # Counterfactual fairness score (lower is better, 0 is perfectly fair)
    cf_fairness_score = final_fairness_penalty
    
    return {
        'counterfactual_fairness_score': cf_fairness_score,
        'predictor_params': theta_opt,
        'latent_variables': U_opt,
        'non_descendants': non_descendant_names,
        'fairness_penalty': final_fairness_penalty,
        'prediction_loss': final_prediction_loss,
        'total_loss': result.fun,
        'converged': result.success,
        'n_iterations': result.nit if hasattr(result, 'nit') else None
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic dataset
    n_samples = 500
    
    # Protected attribute (e.g., gender: 0 or 1)
    A = np.random.binomial(1, 0.5, n_samples)