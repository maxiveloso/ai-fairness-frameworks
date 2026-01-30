import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings

def fairness_accuracy_pareto_frontier_analysis(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    protected_attribute: Union[np.ndarray, pd.Series, str],
    fairness_constraint: str = 'demographic_parity',
    base_classifier: Optional[BaseEstimator] = None,
    n_points: int = 20,
    lambda_range: Tuple[float, float] = (0.0, 1.0),
    scalarization_method: str = 'chebyshev',
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform Fairness-Accuracy Pareto Frontier Analysis using Agarwal et al.'s reductions approach.
    
    This function implements a multi-objective optimization approach to find the trade-off
    between fairness and accuracy in classification tasks. It generates multiple classifiers
    with different fairness-accuracy trade-offs and computes the Pareto frontier.
    
    The reductions approach converts fair classification into a cost-sensitive classification
    problem by introducing fairness constraints as additional costs in the optimization objective.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data features
    y : array-like of shape (n_samples,)
        Binary target variable (0 or 1)
    protected_attribute : array-like of shape (n_samples,) or str
        Protected attribute values or column name if X is DataFrame
    fairness_constraint : str, default='demographic_parity'
        Type of fairness constraint ('demographic_parity', 'equalized_odds', 'equal_opportunity')
    base_classifier : BaseEstimator, optional
        Base classifier to use. If None, uses LogisticRegression
    n_points : int, default=20
        Number of points to generate on the Pareto frontier
    lambda_range : tuple of float, default=(0.0, 1.0)
        Range of lambda values for fairness-accuracy trade-off
    scalarization_method : str, default='chebyshev'
        Method for multi-objective optimization ('linear' or 'chebyshev')
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, optional
        Random state for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'pareto_points': Array of (fairness_violation, accuracy_loss) points
        - 'pareto_classifiers': List of trained classifiers on Pareto frontier
        - 'lambda_values': Lambda values used for each point
        - 'knee_points': Indices of knee points on the frontier
        - 'dominated_points': Points that are dominated by Pareto frontier
        - 'fairness_violations': Fairness constraint violations for each point
        - 'accuracy_scores': Accuracy scores for each point
        - 'optimal_point': Index of optimal point using scalarization
        - 'trade_off_slope': Slope of trade-off curve at each point
        
    Raises
    ------
    ValueError
        If inputs are invalid or incompatible
    """
    
    # Input validation
    X = np.asarray(X) if not isinstance(X, pd.DataFrame) else X
    y = np.asarray(y)
    
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    
    if isinstance(protected_attribute, str):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("protected_attribute as string requires X to be DataFrame")
        protected_attr = X[protected_attribute].values
        X = X.drop(columns=[protected_attribute])
    else:
        protected_attr = np.asarray(protected_attribute)
    
    if len(protected_attr) != len(y):
        raise ValueError("protected_attribute must have same length as y")
    
    if fairness_constraint not in ['demographic_parity', 'equalized_odds', 'equal_opportunity']:
        raise ValueError("Invalid fairness_constraint")
    
    if not 0 <= test_size <= 1:
        raise ValueError("test_size must be between 0 and 1")
    
    # Convert to numpy arrays
    X = np.asarray(X)
    unique_classes = np.unique(y)
    if len(unique_classes) != 2:
        raise ValueError("Only binary classification is supported")
    
    # Split data
    X_train, X_test, y_train, y_test, attr_train, attr_test = train_test_split(
        X, y, protected_attr, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize base classifier
    if base_classifier is None:
        base_classifier = LogisticRegression(random_state=random_state)
    
    # Generate lambda values for trade-off exploration
    lambda_values = np.linspace(lambda_range[0], lambda_range[1], n_points)
    
    # Storage for results
    classifiers = []
    fairness_violations = []
    accuracy_scores = []
    pareto_points = []
    
    def _compute_fairness_violation(y_pred, protected_attr, y_true, constraint_type):
        """Compute fairness constraint violation"""
        groups = np.unique(protected_attr)
        if len(groups) != 2:
            # Handle multi-group case by computing max violation
            violations = []
            for i, group1 in enumerate(groups):
                for group2 in groups[i+1:]:
                    mask1 = protected_attr == group1
                    mask2 = protected_attr == group2
                    
                    if constraint_type == 'demographic_parity':
                        rate1 = np.mean(y_pred[mask1])
                        rate2 = np.mean(y_pred[mask2])
                        violations.append(abs(rate1 - rate2))
                    elif constraint_type == 'equalized_odds':
                        # True positive rate difference
                        tpr1 = np.mean(y_pred[mask1 & (y_true == 1)]) if np.any(mask1 & (y_true == 1)) else 0
                        tpr2 = np.mean(y_pred[mask2 & (y_true == 1)]) if np.any(mask2 & (y_true == 1)) else 0
                        # False positive rate difference  
                        fpr1 = np.mean(y_pred[mask1 & (y_true == 0)]) if np.any(mask1 & (y_true == 0)) else 0
                        fpr2 = np.mean(y_pred[mask2 & (y_true == 0)]) if np.any(mask2 & (y_true == 0)) else 0
                        violations.append(max(abs(tpr1 - tpr2), abs(fpr1 - fpr2)))
                    elif constraint_type == 'equal_opportunity':
                        # True positive rate difference only
                        tpr1 = np.mean(y_pred[mask1 & (y_true == 1)]) if np.any(mask1 & (y_true == 1)) else 0
                        tpr2 = np.mean(y_pred[mask2 & (y_true == 1)]) if np.any(mask2 & (y_true == 1)) else 0
                        violations.append(abs(tpr1 - tpr2))
            return max(violations) if violations else 0.0
        
        # Binary group case
        group1, group2 = groups
        mask1 = protected_attr == group1
        mask2 = protected_attr == group2
        
        if constraint_type == 'demographic_parity':
            rate1 = np.mean(y_pred[mask1])
            rate2 = np.mean(y_pred[mask2])
            return abs(rate1 - rate2)
        elif constraint_type == 'equalized_odds':
            tpr1 = np.mean(y_pred[mask1 & (y_true == 1)]) if np.any(mask1 & (y_true == 1)) else 0
            tpr2 = np.mean(y_pred[mask2 & (y_true == 1)]) if np.any(mask2 & (y_true == 1)) else 0
            fpr1 = np.mean(y_pred[mask1 & (y_true == 0)]) if np.any(mask1 & (y_true == 0)) else 0
            fpr2 = np.mean(y_pred[mask2 & (y_true == 0)]) if np.any(mask2 & (y_true == 0)) else 0
            return max(abs(tpr1 - tpr2), abs(fpr1 - fpr2))
        elif constraint_type == 'equal_opportunity':
            tpr1 = np.mean(y_pred[mask1 & (y_true == 1)]) if np.any(mask1 & (y_true == 1)) else 0
            tpr2 = np.mean(y_pred[mask2 & (y_true == 1)]) if np.any(mask2 & (y_true == 1)) else 0
            return abs(tpr1 - tpr2)
    
    class FairClassifier(BaseEstimator, ClassifierMixin):
        """Wrapper classifier that incorporates fairness constraints"""
        
        def __init__(self, base_classifier, lambda_fair, constraint_type, random_state=None):
            self.base_classifier = base_classifier
            self.lambda_fair = lambda_fair
            self.constraint_type = constraint_type
            self.random_state = random_state
            
        def fit(self, X, y, protected_attr):
            # For simplicity, we use a post-processing approach
            # In practice, Agarwal et al. use more sophisticated reduction methods
            self.base_classifier.fit(X, y)
            
            # Get base predictions
            if hasattr(self.base_classifier, 'predict_proba'):
                base_probs = self.base_classifier.predict_proba(X)[:, 1]
            else:
                base_probs = self.base_classifier.decision_function(X)
                base_probs = 1 / (1 + np.exp(-base_probs))  # sigmoid
            
            # Find optimal threshold that balances accuracy and fairness
            thresholds = np.linspace(0.01, 0.99, 50)
            best_threshold = 0.5
            best_objective = float('inf')
            
            for threshold in thresholds:
                y_pred = (base_probs >= threshold).astype(int)
                accuracy = accuracy_score(y, y_pred)
                fairness_viol = _compute_fairness_violation(y_pred, protected_attr, y, self.constraint_type)
                
                # Combined objective (minimize)
                if scalarization_method == 'chebyshev':
                    objective = max((1 - accuracy), self.lambda_fair * fairness_viol)
                else:  # linear
                    objective = (1 - accuracy) + self.lambda_fair * fairness_viol
                
                if objective < best_objective:
                    best_objective = objective
                    best_threshold = threshold
            
            self.threshold_ = best_