import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

def flip_test(
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    protected_attribute: Union[str, int, np.ndarray],
    model: BaseEstimator,
    source_group: Any = None,
    target_group: Any = None,
    transport_method: str = 'gan',
    gan_epochs: int = 100,
    gan_hidden_size: int = 64,
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform FlipTest fairness analysis using optimal transport mapping.
    
    FlipTest evaluates algorithmic fairness by using optimal transport to translate
    feature distributions between protected groups and measuring prediction changes.
    The method constructs "flipsets" - individuals whose predictions change after
    translation - to quantify discriminatory behavior.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target values (for model training if needed)
    protected_attribute : str, int, or array-like
        Protected attribute column name (if X is DataFrame), column index, or array
    model : sklearn estimator
        Trained model to test for fairness
    source_group : any, optional
        Value identifying source group for transport. If None, uses first unique value
    target_group : any, optional
        Value identifying target group for transport. If None, uses second unique value
    transport_method : str, default='gan'
        Method for optimal transport approximation ('gan' or 'exact')
    gan_epochs : int, default=100
        Number of training epochs for GAN-based transport
    gan_hidden_size : int, default=64
        Hidden layer size for GAN networks
    n_bootstrap : int, default=100
        Number of bootstrap samples for statistical inference
    alpha : float, default=0.05
        Significance level for confidence intervals
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'flip_rate': Proportion of samples that flip predictions
        - 'flip_rate_ci': Confidence interval for flip rate
        - 'flipset_indices': Indices of samples in the flipset
        - 'transport_cost': Cost of optimal transport mapping
        - 'feature_importance': Feature ranking for transport differences
        - 'group_statistics': Statistics for each protected group
        - 'p_value': Statistical significance of observed flip rate
        - 'effect_size': Cohen's d effect size measure
        
    References
    ----------
    Black, E., Yeom, S., & Fredrikson, M. (2020). FlipTest: Fairness testing 
    via optimal transport. In Proceedings of the 2020 Conference on Fairness, 
    Accountability, and Transparency (pp. 111-121).
    """
    
    # Input validation
    if random_state is not None:
        np.random.seed(random_state)
    
    # Convert inputs to appropriate formats
    if isinstance(X, pd.DataFrame):
        if isinstance(protected_attribute, str):
            protected_attr = X[protected_attribute].values
            X_features = X.drop(columns=[protected_attribute]).values
            feature_names = X.drop(columns=[protected_attribute]).columns.tolist()
        else:
            protected_attr = X.iloc[:, protected_attribute].values
            X_features = X.drop(X.columns[protected_attribute], axis=1).values
            feature_names = X.drop(X.columns[protected_attribute], axis=1).columns.tolist()
    else:
        X = np.asarray(X)
        if isinstance(protected_attribute, np.ndarray):
            protected_attr = protected_attribute
        else:
            protected_attr = X[:, protected_attribute]
            X_features = np.delete(X, protected_attribute, axis=1)
        feature_names = [f'feature_{i}' for i in range(X_features.shape[1])]
    
    y = np.asarray(y)
    
    # Validate inputs
    if len(X_features) != len(y) or len(X_features) != len(protected_attr):
        raise ValueError("X, y, and protected_attribute must have same length")
    
    if transport_method not in ['gan', 'exact']:
        raise ValueError("transport_method must be 'gan' or 'exact'")
    
    # Identify protected groups
    unique_groups = np.unique(protected_attr)
    if len(unique_groups) < 2:
        raise ValueError("Protected attribute must have at least 2 groups")
    
    if source_group is None:
        source_group = unique_groups[0]
    if target_group is None:
        target_group = unique_groups[1] if len(unique_groups) > 1 else unique_groups[0]
    
    # Extract group data
    source_mask = protected_attr == source_group
    target_mask = protected_attr == target_group
    
    X_source = X_features[source_mask]
    X_target = X_features[target_mask]
    
    if len(X_source) == 0 or len(X_target) == 0:
        raise ValueError("Both source and target groups must have samples")
    
    # Standardize features for transport
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    # Perform optimal transport mapping
    if transport_method == 'gan':
        X_source_transported, transport_cost = _gan_transport(
            X_source_scaled, X_target_scaled, gan_epochs, gan_hidden_size, random_state
        )
    else:
        X_source_transported, transport_cost = _exact_transport(
            X_source_scaled, X_target_scaled
        )
    
    # Transform back to original scale
    X_source_transported = scaler.inverse_transform(X_source_transported)
    
    # Get original predictions for source group
    original_predictions = model.predict(X_source)
    
    # Get predictions after transport
    transported_predictions = model.predict(X_source_transported)
    
    # Identify flipset based on prediction changes
    if hasattr(model, "predict_proba"):
        # For classification, check if predicted class changes
        original_classes = model.predict(X_source)
        transported_classes = model.predict(X_source_transported)
        flip_mask = original_classes != transported_classes
    else:
        # For regression, check if prediction changes significantly
        prediction_diff = np.abs(transported_predictions - original_predictions)
        threshold = np.std(original_predictions) * 0.1  # 10% of std as threshold
        flip_mask = prediction_diff > threshold
    
    flipset_indices = np.where(source_mask)[0][flip_mask]
    flip_rate = np.mean(flip_mask)
    
    # Bootstrap confidence interval for flip rate
    bootstrap_flip_rates = []
    for _ in range(n_bootstrap):
        boot_indices = np.random.choice(len(X_source), len(X_source), replace=True)
        boot_flip_rate = np.mean(flip_mask[boot_indices])
        bootstrap_flip_rates.append(boot_flip_rate)
    
    flip_rate_ci = np.percentile(bootstrap_flip_rates, [100 * alpha/2, 100 * (1 - alpha/2)])
    
    # Statistical significance test (one-sample t-test against null hypothesis of no flips)
    t_stat, p_value = stats.ttest_1samp(bootstrap_flip_rates, 0)
    
    # Effect size (Cohen's d)
    effect_size = np.mean(bootstrap_flip_rates) / np.std(bootstrap_flip_rates) if np.std(bootstrap_flip_rates) > 0 else 0
    
    # Feature importance analysis
    feature_importance = _compute_feature_importance(
        X_source_scaled, X_source_transported, feature_names
    )
    
    # Group statistics
    group_stats = {
        'source_group': {
            'group_value': source_group,
            'n_samples': len(X_source),
            'mean_prediction': np.mean(original_predictions),
            'std_prediction': np.std(original_predictions)
        },
        'target_group': {
            'group_value': target_group,
            'n_samples': len(X_target),
            'mean_prediction': np.mean(model.predict(X_target)),
            'std_prediction': np.std(model.predict(X_target))
        }
    }
    
    return {
        'flip_rate': flip_rate,
        'flip_rate_ci': flip_rate_ci,
        'flipset_indices': flipset_indices,
        'transport_cost': transport_cost,
        'feature_importance': feature_importance,
        'group_statistics': group_stats,
        'p_value': p_value,
        'effect_size': effect_size,
        'n_flips': int(np.sum(flip_mask)),
        'total_source_samples': len(X_source)
    }


def _gan_transport(X_source: np.ndarray, X_target: np.ndarray, 
                  epochs: int, hidden_size: int, random_state: Optional[int]) -> Tuple[np.ndarray, float]:
    """
    Approximate optimal transport using GAN-based approach.
    
    This implements a simplified version of GAN-based optimal transport
    using neural networks to learn the mapping between distributions.
    """
    
    # Simple neural network-based transport approximation
    # In practice, this would use more sophisticated GAN architecture
    transport_model = MLPRegressor(
        hidden_layer_sizes=(hidden_size, hidden_size),
        max_iter=epochs,
        random_state=random_state,
        alpha=0.01
    )
    
    # Create synthetic target samples for training
    n_samples = min(len(X_source), len(X_target))
    source_sample_idx = np.random.choice(len(X_source), n_samples, replace=False)
    target_sample_idx = np.random.choice(len(X_target), n_samples, replace=False)
    
    X_source_sample = X_source[source_sample_idx]
    X_target_sample = X_target[target_sample_idx]
    
    # Train transport mapping
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transport_model.fit(X_source_sample, X_target_sample)
    
    # Apply transport to all source samples
    X_source_transported = transport_model.predict(X_source)
    
    # Compute approximate transport cost
    transport_cost = np.mean(np.sum((X_source - X_source_transported) ** 2, axis=1))
    
    return X_source_transported, transport_cost


def _exact_transport(X_source: np.ndarray, X_target: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute exact optimal transport using linear assignment.
    
    This is a simplified version that works for small datasets.
    In practice, specialized optimal transport libraries would be used.
    """
    
    # Compute pairwise distances
    distances = cdist(X_source, X_target, metric='