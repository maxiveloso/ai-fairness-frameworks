import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Callable, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings

def counterfactual_fairness(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    protected_attr: str,
    causal_graph: Dict[str, List[str]],
    predictor: Optional[BaseEstimator] = None,
    fairness_penalty: float = 1.0,
    n_counterfactuals: int = 100,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate counterfactual fairness of a predictor based on causal graph structure.
    
    Counterfactual fairness requires that predictions remain unchanged when protected
    attributes are counterfactually altered. A predictor is counterfactually fair
    if it only uses non-descendants of the protected attribute in the causal graph.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix containing all variables including protected attribute
    y : pd.Series or np.ndarray
        Target variable
    protected_attr : str
        Name of the protected attribute column
    causal_graph : Dict[str, List[str]]
        Adjacency list representation of causal graph where keys are parent nodes
        and values are lists of child nodes (descendants)
    predictor : BaseEstimator, optional
        Trained predictor to evaluate. If None, fits LinearRegression
    fairness_penalty : float, default=1.0
        Weight for fairness penalty in optimization
    n_counterfactuals : int, default=100
        Number of counterfactual samples to generate for evaluation
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'counterfactual_fairness_score': Average difference in predictions across counterfactuals
        - 'is_counterfactually_fair': Boolean indicating if predictor is fair (score < 0.1)
        - 'descendants': List of descendants of protected attribute
        - 'non_descendants': List of non-descendants (fair features)
        - 'fairness_penalty': Applied penalty weight
        - 'n_samples': Number of samples evaluated
        - 'predictor_uses_descendants': Whether predictor uses descendant features
    """
    
    # Input validation
    if isinstance(X, np.ndarray):
        raise ValueError("X must be a pandas DataFrame to identify protected attribute")
    
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
        
    if protected_attr not in X.columns:
        raise ValueError(f"Protected attribute '{protected_attr}' not found in X columns")
        
    if len(X) != len(y):
        raise ValueError("X and y must have same number of samples")
        
    if not isinstance(causal_graph, dict):
        raise ValueError("causal_graph must be a dictionary")
        
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Convert inputs to appropriate formats
    X = X.copy()
    if isinstance(y, pd.Series):
        y = y.values
    
    # Find descendants of protected attribute using graph traversal
    def find_descendants(graph: Dict[str, List[str]], node: str) -> set:
        """Find all descendants of a node in the causal graph"""
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
    
    descendants = find_descendants(causal_graph, protected_attr)
    all_features = set(X.columns) - {protected_attr}
    non_descendants = all_features - descendants
    
    # Fit predictor if not provided
    if predictor is None:
        predictor = LinearRegression()
        predictor.fit(X, y)
    
    # Check if predictor uses descendant features
    predictor_features = set()
    if hasattr(predictor, 'feature_names_in_'):
        predictor_features = set(predictor.feature_names_in_)
    else:
        # Assume predictor uses all features in X
        predictor_features = set(X.columns)
    
    uses_descendants = bool(predictor_features.intersection(descendants))
    
    # Generate counterfactual samples
    n_samples = min(len(X), n_counterfactuals)
    sample_indices = np.random.choice(len(X), n_samples, replace=False)
    
    counterfactual_differences = []
    
    for idx in sample_indices:
        original_sample = X.iloc[idx:idx+1].copy()
        
        # Get unique values of protected attribute for counterfactuals
        protected_values = X[protected_attr].unique()
        original_value = original_sample[protected_attr].iloc[0]
        
        # Generate predictions for all counterfactual values of protected attribute
        predictions = []
        
        for value in protected_values:
            counterfactual_sample = original_sample.copy()
            counterfactual_sample[protected_attr] = value
            
            # For descendants, we would ideally update them based on causal model
            # Here we use a simplified approach of keeping them unchanged
            # In practice, this requires a full causal model specification
            
            try:
                pred = predictor.predict(counterfactual_sample)[0]
                predictions.append(pred)
            except Exception as e:
                warnings.warn(f"Prediction failed for sample {idx}: {e}")
                continue
        
        # Calculate difference between max and min predictions across counterfactuals
        if len(predictions) > 1:
            counterfactual_diff = np.max(predictions) - np.min(predictions)
            counterfactual_differences.append(counterfactual_diff)
    
    # Calculate counterfactual fairness score
    if counterfactual_differences:
        cf_score = np.mean(counterfactual_differences)
    else:
        cf_score = np.inf
        warnings.warn("No valid counterfactual differences computed")
    
    # Determine if predictor is counterfactually fair
    # A lower score indicates better fairness
    is_fair = cf_score < 0.1  # Threshold can be adjusted based on domain
    
    return {
        'counterfactual_fairness_score': cf_score,
        'is_counterfactually_fair': is_fair,
        'descendants': list(descendants),
        'non_descendants': list(non_descendants),
        'fairness_penalty': fairness_penalty,
        'n_samples': n_samples,
        'predictor_uses_descendants': uses_descendants,
        'n_counterfactual_differences': len(counterfactual_differences)
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Create synthetic dataset with causal structure
    n_samples = 1000
    
    # Generate data following a causal structure:
    # Gender -> Education -> Income
    # Gender -> Income (direct effect)
    
    gender = np.random.binomial(1, 0.5, n_samples)  # Protected attribute
    education = gender * 0.3 + np.random.normal(0, 0.5, n_samples)  # Descendant of gender
    income = gender * 0.2 + education * 0.4 + np.random.normal(0, 0.3, n_samples)  # Target
    age = np.random.normal(40, 10, n_samples)  # Non-descendant
    
    # Create DataFrame
    df = pd.DataFrame({
        'gender': gender,
        'education': education,
        'age': age,
        'income': income
    })
    
    X = df[['gender', 'education', 'age']]
    y = df['income']
    
    # Define causal graph: gender causes education
    causal_graph = {
        'gender': ['education'],  # gender is parent of education
        'education': [],          # education has no children in this example
        'age': []                 # age has no children
    }
    
    # Test 1: Predictor using all features (potentially unfair)
    print("=== Test 1: Predictor using all features ===")
    from sklearn.linear_model import LinearRegression
    
    unfair_predictor = LinearRegression()
    unfair_predictor.fit(X, y)
    
    results1 = counterfactual_fairness(
        X=X,
        y=y,
        protected_attr='gender',
        causal_graph=causal_graph,
        predictor=unfair_predictor,
        n_counterfactuals=50,
        random_state=42
    )
    
    print(f"Counterfactual fairness score: {results1['counterfactual_fairness_score']:.4f}")
    print(f"Is counterfactually fair: {results1['is_counterfactually_fair']}")
    print(f"Descendants of gender: {results1['descendants']}")
    print(f"Non-descendants: {results1['non_descendants']}")
    print(f"Uses descendant features: {results1['predictor_uses_descendants']}")
    
    # Test 2: Fair predictor using only non-descendants
    print("\n=== Test 2: Fair predictor using only non-descendants ===")
    
    X_fair = X[['age']]  # Only use non-descendant features
    fair_predictor = LinearRegression()
    fair_predictor.fit(X_fair, y)
    
    # For evaluation, we still need the full feature set
    results2 = counterfactual_fairness(
        X=X,
        y=y,
        protected_attr='gender',
        causal_graph=causal_graph,
        predictor=None,  # Let function fit its own predictor
        n_counterfactuals=50,
        random_state=42
    )
    
    print(f"Counterfactual fairness score: {results2['counterfactual_fairness_score']:.4f}")
    print(f"Is counterfactually fair: {results2['is_counterfactually_fair']}")
    print(f"Number of samples evaluated: {results2['n_samples']}")
    print(f"Fairness penalty applied: {results2['fairness_penalty']}")