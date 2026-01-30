import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings

def cost_proportionate_example_weighting(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    costs: Union[np.ndarray, pd.Series, Dict[int, float]],
    method: str = 'weighting',
    base_estimator: Optional[BaseEstimator] = None,
    n_estimators: int = 10,
    random_state: Optional[int] = None,
    normalize_weights: bool = True
) -> Dict[str, Any]:
    """
    Implement cost-proportionate example weighting for cost-sensitive learning.
    
    This technique addresses the problem of misclassification costs by either:
    1. Direct weighting: Assigns weights proportional to misclassification costs
    2. Rejection sampling: Creates cost-proportionate samples via rejection sampling
    
    The core idea is that examples with higher misclassification costs should have
    greater influence on the learning algorithm, either through explicit weighting
    or through increased sampling probability.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training feature matrix
    y : array-like of shape (n_samples,)
        Training target labels
    costs : array-like or dict
        Misclassification costs. If array-like, should have same length as y.
        If dict, maps class labels to costs.
    method : str, default='weighting'
        Method to use: 'weighting' for direct weighting, 'sampling' for rejection sampling
    base_estimator : estimator, default=None
        Base estimator for ensemble methods. If None, uses DecisionTreeClassifier
    n_estimators : int, default=10
        Number of estimators for ensemble method (only used with 'sampling')
    random_state : int, default=None
        Random state for reproducibility
    normalize_weights : bool, default=True
        Whether to normalize weights to sum to number of samples
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'method': Method used ('weighting' or 'sampling')
        - 'sample_weights': Sample weights (for weighting method)
        - 'resampled_indices': Indices of resampled data (for sampling method)
        - 'cost_distribution': Distribution of costs across classes
        - 'effective_sample_sizes': Effective sample sizes per class
        - 'weight_statistics': Statistics about the computed weights
    """
    
    # Input validation
    X, y = check_X_y(X, y, accept_sparse=False)
    
    if method not in ['weighting', 'sampling']:
        raise ValueError("method must be 'weighting' or 'sampling'")
    
    # Convert costs to appropriate format
    if isinstance(costs, dict):
        # Map class costs to sample costs
        sample_costs = np.array([costs.get(label, 1.0) for label in y])
    else:
        costs = check_array(costs, ensure_2d=False)
        if len(costs) != len(y):
            raise ValueError("costs array must have same length as y")
        sample_costs = np.array(costs)
    
    # Ensure all costs are positive
    if np.any(sample_costs <= 0):
        raise ValueError("All costs must be positive")
    
    classes = unique_labels(y)
    n_samples = len(y)
    
    # Calculate cost distribution across classes
    cost_distribution = {}
    for cls in classes:
        mask = y == cls
        cost_distribution[cls] = {
            'mean_cost': np.mean(sample_costs[mask]),
            'total_cost': np.sum(sample_costs[mask]),
            'sample_count': np.sum(mask)
        }
    
    if method == 'weighting':
        # Direct weighting approach
        # Weights are proportional to misclassification costs
        sample_weights = sample_costs.copy()
        
        if normalize_weights:
            # Normalize weights to sum to number of samples
            sample_weights = sample_weights * (n_samples / np.sum(sample_weights))
        
        # Calculate effective sample sizes
        effective_sample_sizes = {}
        for cls in classes:
            mask = y == cls
            effective_sample_sizes[cls] = np.sum(sample_weights[mask])
        
        # Weight statistics
        weight_stats = {
            'min_weight': np.min(sample_weights),
            'max_weight': np.max(sample_weights),
            'mean_weight': np.mean(sample_weights),
            'std_weight': np.std(sample_weights),
            'weight_ratio': np.max(sample_weights) / np.min(sample_weights)
        }
        
        return {
            'method': 'weighting',
            'sample_weights': sample_weights,
            'resampled_indices': None,
            'cost_distribution': cost_distribution,
            'effective_sample_sizes': effective_sample_sizes,
            'weight_statistics': weight_stats
        }
    
    else:  # method == 'sampling'
        # Rejection sampling approach
        np.random.seed(random_state)
        
        # Calculate sampling probabilities proportional to costs
        sampling_probs = sample_costs / np.sum(sample_costs)
        
        # Perform rejection sampling to create resampled dataset
        # Sample size is typically kept same as original for each bootstrap
        resampled_indices_list = []
        
        for _ in range(n_estimators):
            # Sample with replacement according to cost-proportionate probabilities
            bootstrap_indices = np.random.choice(
                n_samples, 
                size=n_samples, 
                replace=True, 
                p=sampling_probs
            )
            resampled_indices_list.append(bootstrap_indices)
        
        # Calculate effective sample sizes based on average sampling frequency
        all_indices = np.concatenate(resampled_indices_list)
        unique_indices, counts = np.unique(all_indices, return_counts=True)
        avg_sample_freq = counts / n_estimators
        
        effective_sample_sizes = {}
        for cls in classes:
            mask = y == cls
            class_indices = np.where(mask)[0]
            # Find intersection and sum frequencies
            intersect_mask = np.isin(unique_indices, class_indices)
            effective_sample_sizes[cls] = np.sum(avg_sample_freq[intersect_mask])
        
        # Sampling statistics
        sampling_stats = {
            'avg_samples_per_bootstrap': n_samples,
            'total_bootstraps': n_estimators,
            'unique_samples_used': len(unique_indices),
            'max_sample_frequency': np.max(counts),
            'min_sample_frequency': np.min(counts)
        }
        
        return {
            'method': 'sampling',
            'sample_weights': None,
            'resampled_indices': resampled_indices_list,
            'cost_distribution': cost_distribution,
            'effective_sample_sizes': effective_sample_sizes,
            'weight_statistics': sampling_stats
        }


class CostProportionateClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier wrapper that implements cost-proportionate example weighting.
    
    This classifier can use either direct weighting or rejection sampling
    to handle cost-sensitive learning problems.
    """
    
    def __init__(self, base_estimator=None, method='weighting', 
                 n_estimators=10, random_state=None, normalize_weights=True):
        self.base_estimator = base_estimator
        self.method = method
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.normalize_weights = normalize_weights
    
    def fit(self, X, y, costs):
        """Fit the cost-proportionate classifier."""
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        
        # Get cost-proportionate weighting/sampling
        result = cost_proportionate_example_weighting(
            X, y, costs, 
            method=self.method,
            base_estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            normalize_weights=self.normalize_weights
        )
        
        self.cost_result_ = result
        
        # Initialize base estimator
        if self.base_estimator is None:
            base_est = DecisionTreeClassifier(random_state=self.random_state)
        else:
            base_est = self.base_estimator
        
        if self.method == 'weighting':
            # Train single estimator with sample weights
            self.estimator_ = base_est
            self.estimator_.fit(X, y, sample_weight=result['sample_weights'])
        else:
            # Train ensemble with resampled data
            self.estimators_ = []
            for indices in result['resampled_indices']:
                est = base_est.__class__(**base_est.get_params())
                est.fit(X[indices], y[indices])
                self.estimators_.append(est)
        
        return self
    
    def predict(self, X):
        """Predict using the fitted cost-proportionate classifier."""
        X = check_array(X)
        
        if self.method == 'weighting':
            return self.estimator_.predict(X)
        else:
            # Ensemble prediction via majority voting
            predictions = np.array([est.predict(X) for est in self.estimators_])
            # Simple majority vote
            return np.array([np.bincount(predictions[:, i]).argmax() 
                           for i in range(X.shape[0])])


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, 
                             n_redundant=0, random_state=42)
    
    # Create cost-sensitive scenario where class 2 has higher misclassification cost
    costs = np.ones(len(y))
    costs[y == 2] = 5.0  # Class 2 has 5x higher misclassification cost
    costs[y == 1] = 2.0  # Class 1 has 2x higher cost
    
    X_train, X_test, y_train, y_test, costs_train, costs_test = train_test_split(
        X, y, costs, test_size=0.3, random_state=42, stratify=y
    )
    
    print("Cost-Proportionate Example Weighting Example")
    print("=" * 50)
    
    # Test weighting method
    print("\n1. Direct Weighting Method:")
    result_weight = cost_proportionate_example_weighting(
        X_train, y_train, costs_train, method='weighting'
    )
    
    print(f"Method: {result_weight['method']}")
    print(f"Weight statistics: {result_weight['weight_statistics']}")
    print(f"Cost distribution: {result_weight['cost_distribution']}")
    print(f"Effective sample sizes: {result_weight['effective_sample_sizes']}")
    
    # Test sampling method
    print("\n2. Rejection Sampling Method:")
    result_sample = cost_proportionate_example_weighting