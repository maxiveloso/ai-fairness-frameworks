import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class FairnessMetrics:
    """Container for fairness metrics"""
    demographic_parity: float
    equalized_odds: float
    equal_opportunity: float
    calibration: float

def pareto_frontier_fairness_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    sensitive_attr: np.ndarray,
    fairness_weights: Optional[List[float]] = None,
    base_classifier: Optional[BaseEstimator] = None,
    fairness_constraint: str = 'demographic_parity',
    threshold_optimization: bool = True,
    cv_folds: int = 5,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, List, plt.Figure]]:
    """
    Perform Pareto Frontier Analysis for Fairness-Accuracy Trade-offs.
    
    This implementation follows Menon & Williamson (2018) approach using cost-sensitive
    risk formulation with instance-dependent thresholding. The method trains multiple
    models with varying fairness constraint weights and identifies Pareto-optimal
    points representing optimal fairness-accuracy trade-offs.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Binary target variable (0 or 1) of shape (n_samples,)
    sensitive_attr : np.ndarray
        Binary sensitive attribute (0 or 1) of shape (n_samples,)
    fairness_weights : List[float], optional
        Weights for fairness constraints. If None, uses logarithmic spacing
    base_classifier : BaseEstimator, optional
        Base classifier to use. If None, uses LogisticRegression
    fairness_constraint : str, default='demographic_parity'
        Type of fairness constraint ('demographic_parity', 'equalized_odds', 'equal_opportunity')
    threshold_optimization : bool, default=True
        Whether to optimize decision thresholds for each group
    cv_folds : int, default=5
        Number of cross-validation folds for model evaluation
    random_state : int, optional
        Random state for reproducibility
        
    Returns
    -------
    Dict containing:
        - 'pareto_points': Array of Pareto-optimal (fairness_violation, accuracy) points
        - 'all_points': Array of all (fairness_violation, accuracy) points
        - 'pareto_indices': Indices of Pareto-optimal points
        - 'price_of_fairness': Price of Fairness metric
        - 'optimal_thresholds': Optimal thresholds for each fairness weight
        - 'fairness_metrics': Detailed fairness metrics for each model
        - 'models': Trained models for each fairness weight
        - 'frontier_plot': Matplotlib figure showing the Pareto frontier
    """
    
    # Input validation
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array")
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError("y must be a 1D numpy array")
    if not isinstance(sensitive_attr, np.ndarray) or sensitive_attr.ndim != 1:
        raise ValueError("sensitive_attr must be a 1D numpy array")
    if len(X) != len(y) or len(X) != len(sensitive_attr):
        raise ValueError("X, y, and sensitive_attr must have the same number of samples")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only binary values (0, 1)")
    if not np.all(np.isin(sensitive_attr, [0, 1])):
        raise ValueError("sensitive_attr must contain only binary values (0, 1)")
    
    # Set default parameters
    if fairness_weights is None:
        fairness_weights = np.logspace(-3, 2, 20).tolist()
    if base_classifier is None:
        base_classifier = LogisticRegression(random_state=random_state)
    
    n_samples, n_features = X.shape
    n_weights = len(fairness_weights)
    
    # Initialize storage for results
    accuracies = np.zeros(n_weights)
    fairness_violations = np.zeros(n_weights)
    optimal_thresholds = []
    fairness_metrics_list = []
    trained_models = []
    
    # Identify privileged and unprivileged groups
    privileged_mask = sensitive_attr == 1
    unprivileged_mask = sensitive_attr == 0
    
    def _compute_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_prob: np.ndarray, sensitive: np.ndarray) -> FairnessMetrics:
        """Compute comprehensive fairness metrics"""
        priv_mask = sensitive == 1
        unpriv_mask = sensitive == 0
        
        # Demographic Parity: P(Y_hat=1|A=1) - P(Y_hat=1|A=0)
        dp_priv = np.mean(y_pred[priv_mask])
        dp_unpriv = np.mean(y_pred[unpriv_mask])
        demographic_parity = abs(dp_priv - dp_unpriv)
        
        # Equalized Odds: max over y in {0,1} of |P(Y_hat=1|A=1,Y=y) - P(Y_hat=1|A=0,Y=y)|
        eo_y1_priv = np.mean(y_pred[priv_mask & (y_true == 1)]) if np.any(priv_mask & (y_true == 1)) else 0
        eo_y1_unpriv = np.mean(y_pred[unpriv_mask & (y_true == 1)]) if np.any(unpriv_mask & (y_true == 1)) else 0
        eo_y0_priv = np.mean(y_pred[priv_mask & (y_true == 0)]) if np.any(priv_mask & (y_true == 0)) else 0
        eo_y0_unpriv = np.mean(y_pred[unpriv_mask & (y_true == 0)]) if np.any(unpriv_mask & (y_true == 0)) else 0
        equalized_odds = max(abs(eo_y1_priv - eo_y1_unpriv), abs(eo_y0_priv - eo_y0_unpriv))
        
        # Equal Opportunity: |P(Y_hat=1|A=1,Y=1) - P(Y_hat=1|A=0,Y=1)|
        equal_opportunity = abs(eo_y1_priv - eo_y1_unpriv)
        
        # Calibration: |P(Y=1|Y_hat=1,A=1) - P(Y=1|Y_hat=1,A=0)|
        cal_priv = np.mean(y_true[priv_mask & (y_pred == 1)]) if np.any(priv_mask & (y_pred == 1)) else 0
        cal_unpriv = np.mean(y_true[unpriv_mask & (y_pred == 1)]) if np.any(unpriv_mask & (y_pred == 1)) else 0
        calibration = abs(cal_priv - cal_unpriv)
        
        return FairnessMetrics(demographic_parity, equalized_odds, equal_opportunity, calibration)
    
    def _optimize_thresholds(y_prob: np.ndarray, y_true: np.ndarray, 
                           sensitive: np.ndarray, fairness_weight: float) -> Tuple[float, float]:
        """Optimize decision thresholds for each group using cost-sensitive approach"""
        
        def cost_function(thresholds):
            t_priv, t_unpriv = thresholds
            
            # Generate predictions using group-specific thresholds
            y_pred = np.zeros_like(y_true)
            priv_mask = sensitive == 1
            unpriv_mask = sensitive == 0
            
            y_pred[priv_mask] = (y_prob[priv_mask] >= t_priv).astype(int)
            y_pred[unpriv_mask] = (y_prob[unpriv_mask] >= t_unpriv).astype(int)
            
            # Compute accuracy (negative for minimization)
            accuracy = accuracy_score(y_true, y_pred)
            
            # Compute fairness violation based on constraint type
            if fairness_constraint == 'demographic_parity':
                dp_priv = np.mean(y_pred[priv_mask])
                dp_unpriv = np.mean(y_pred[unpriv_mask])
                fairness_viol = abs(dp_priv - dp_unpriv)
            elif fairness_constraint == 'equal_opportunity':
                pos_mask = y_true == 1
                eo_priv = np.mean(y_pred[priv_mask & pos_mask]) if np.any(priv_mask & pos_mask) else 0
                eo_unpriv = np.mean(y_pred[unpriv_mask & pos_mask]) if np.any(unpriv_mask & pos_mask) else 0
                fairness_viol = abs(eo_priv - eo_unpriv)
            else:  # equalized_odds
                metrics = _compute_fairness_metrics(y_true, y_pred, y_prob, sensitive)
                fairness_viol = metrics.equalized_odds
            
            # Cost-sensitive objective: minimize accuracy loss + fairness_weight * fairness_violation
            return -accuracy + fairness_weight * fairness_viol
        
        # Grid search over threshold combinations
        threshold_range = np.linspace(0.1, 0.9, 20)
        best_cost = float('inf')
        best_thresholds = (0.5, 0.5)
        
        for t_priv in threshold_range:
            for t_unpriv in threshold_range:
                cost = cost_function((t_priv, t_unpriv))
                if cost < best_cost:
                    best_cost = cost
                    best_thresholds = (t_priv, t_unpriv)
        
        return best_thresholds
    
    # Train models with different fairness constraint weights
    for i, weight in enumerate(fairness_weights):
        # Clone base classifier
        model = type(base_classifier)(**base_classifier.get_params())
        
        # Train model
        model.fit(X, y)
        trained_models.append(model)
        
        # Get probability predictions
        y_prob = model.predict_proba(X)[:, 1]
        
        if threshold_optimization:
            # Optimize thresholds for fairness-accuracy trade-off
            thresh_priv, thresh_unpriv = _optimize_thresholds(y_prob, y, sensitive_attr, weight)
            optimal_thresholds.append((thresh_priv, thresh_unpriv))
            
            # Generate predictions using optimized thresholds
            y_pred = np.zeros_like(y)