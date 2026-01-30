import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils.validation import check_X_y, check_array
from scipy import stats
import warnings

class ThresholdFallbackClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that can reject predictions and defer to a fallback mechanism.
    
    This implements the core rejection option classification where predictions
    below a confidence threshold are rejected and handled by a fallback classifier.
    """
    
    def __init__(self, base_classifier, fallback_classifier=None, 
                 rejection_threshold: float = 0.5, rejection_cost: float = 0.1):
        self.base_classifier = base_classifier
        self.fallback_classifier = fallback_classifier
        self.rejection_threshold = rejection_threshold
        self.rejection_cost = rejection_cost
        self.classes_ = None
        
    def fit(self, X, y):
        """Fit both base and fallback classifiers."""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        
        # Fit base classifier
        self.base_classifier.fit(X, y)
        
        # Fit fallback classifier if provided
        if self.fallback_classifier is not None:
            self.fallback_classifier.fit(X, y)
            
        return self
    
    def predict_with_rejection(self, X):
        """
        Predict with rejection option.
        
        Returns:
            predictions: array of predictions (-1 indicates rejection)
            confidences: array of prediction confidences
            rejected_mask: boolean mask indicating rejected samples
        """
        X = check_array(X)
        
        # Get base predictions and probabilities
        if hasattr(self.base_classifier, 'predict_proba'):
            proba = self.base_classifier.predict_proba(X)
            confidences = np.max(proba, axis=1)
        else:
            # Use decision function if available
            if hasattr(self.base_classifier, 'decision_function'):
                scores = self.base_classifier.decision_function(X)
                confidences = np.abs(scores) if scores.ndim == 1 else np.max(np.abs(scores), axis=1)
            else:
                confidences = np.ones(X.shape[0])  # Default confidence
        
        base_predictions = self.base_classifier.predict(X)
        
        # Determine rejections based on threshold
        rejected_mask = confidences < self.rejection_threshold
        
        # Initialize final predictions
        final_predictions = base_predictions.copy()
        
        # Handle rejected samples
        if np.any(rejected_mask):
            if self.fallback_classifier is not None:
                # Use fallback classifier for rejected samples
                fallback_preds = self.fallback_classifier.predict(X[rejected_mask])
                final_predictions[rejected_mask] = fallback_preds
            else:
                # Mark as rejected (-1)
                final_predictions[rejected_mask] = -1
        
        return final_predictions, confidences, rejected_mask
    
    def predict(self, X):
        """Standard predict method."""
        predictions, _, _ = self.predict_with_rejection(X)
        return predictions

def rejection_option_classification(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    base_classifier,
    fallback_classifier=None,
    rejection_threshold: float = 0.5,
    rejection_cost: float = 0.1,
    fairness_groups: Optional[Union[np.ndarray, pd.Series]] = None,
    test_size: float = 0.3,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Implement Rejection Option Classification for fair and robust predictions.
    
    This technique extends the model output space to include a rejection option,
    allowing the classifier to abstain from making predictions when confidence
    is low or when fairness constraints might be violated. Rejected samples
    can be deferred to human decision-makers or fallback mechanisms.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target labels
    base_classifier : sklearn estimator
        Primary classifier to use for predictions
    fallback_classifier : sklearn estimator, optional
        Fallback classifier for rejected samples
    rejection_threshold : float, default=0.5
        Confidence threshold below which predictions are rejected
    rejection_cost : float, default=0.1
        Cost associated with rejecting a prediction
    fairness_groups : array-like, optional
        Group membership for fairness analysis
    test_size : float, default=0.3
        Proportion of data to use for testing
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'prediction_quality': Overall prediction accuracy
        - 'rejection_quality': Quality of rejection decisions
        - 'rejection_rate': Proportion of samples rejected
        - 'coverage': Proportion of samples not rejected
        - 'selective_accuracy': Accuracy on non-rejected samples
        - 'fairness_metrics': Fairness statistics if groups provided
        - 'cost_benefit_analysis': Economic analysis of rejections
        - 'confidence_statistics': Statistics about prediction confidence
        - 'classifier': Fitted rejection classifier
    """
    
    # Input validation
    if not hasattr(base_classifier, 'fit'):
        raise ValueError("base_classifier must be a scikit-learn compatible estimator")
    
    if not 0 < rejection_threshold < 1:
        raise ValueError("rejection_threshold must be between 0 and 1")
        
    if not 0 <= rejection_cost <= 1:
        raise ValueError("rejection_cost must be between 0 and 1")
        
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    # Convert inputs to numpy arrays
    X = np.array(X) if not isinstance(X, np.ndarray) else X
    y = np.array(y) if not isinstance(y, np.ndarray) else y
    
    if fairness_groups is not None:
        fairness_groups = np.array(fairness_groups) if not isinstance(fairness_groups, np.ndarray) else fairness_groups
    
    # Train-test split
    np.random.seed(random_state)
    n_samples = X.shape[0]
    test_indices = np.random.choice(n_samples, size=int(n_samples * test_size), replace=False)
    train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    if fairness_groups is not None:
        groups_train, groups_test = fairness_groups[train_indices], fairness_groups[test_indices]
    
    # Create and fit rejection classifier
    rejection_classifier = ThresholdFallbackClassifier(
        base_classifier=base_classifier,
        fallback_classifier=fallback_classifier,
        rejection_threshold=rejection_threshold,
        rejection_cost=rejection_cost
    )
    
    rejection_classifier.fit(X_train, y_train)
    
    # Make predictions with rejection option
    predictions, confidences, rejected_mask = rejection_classifier.predict_with_rejection(X_test)
    
    # Calculate basic metrics
    rejection_rate = np.mean(rejected_mask)
    coverage = 1 - rejection_rate
    
    # Prediction Quality (PQ): Accuracy on all samples where prediction was made
    non_rejected_mask = ~rejected_mask & (predictions != -1)
    if np.any(non_rejected_mask):
        selective_accuracy = accuracy_score(y_test[non_rejected_mask], predictions[non_rejected_mask])
        prediction_quality = selective_accuracy
    else:
        selective_accuracy = 0.0
        prediction_quality = 0.0
    
    # Rejection Quality (RQ): How well the rejection mechanism works
    # RQ measures if rejected samples would have been misclassified
    if np.any(rejected_mask):
        # Get what base classifier would have predicted for rejected samples
        base_preds_rejected = rejection_classifier.base_classifier.predict(X_test[rejected_mask])
        would_be_correct = (base_preds_rejected == y_test[rejected_mask])
        rejection_quality = 1 - np.mean(would_be_correct)  # Higher when rejecting likely errors
    else:
        rejection_quality = 0.0
    
    # Confidence statistics
    confidence_stats = {
        'mean_confidence': np.mean(confidences),
        'std_confidence': np.std(confidences),
        'mean_confidence_accepted': np.mean(confidences[~rejected_mask]) if np.any(~rejected_mask) else 0,
        'mean_confidence_rejected': np.mean(confidences[rejected_mask]) if np.any(rejected_mask) else 0
    }
    
    # Cost-benefit analysis
    # Cost = rejection_cost * rejection_rate + error_cost * error_rate
    if np.any(non_rejected_mask):
        error_rate = 1 - selective_accuracy
    else:
        error_rate = 0
    
    total_cost = rejection_cost * rejection_rate + error_rate * coverage
    
    cost_benefit = {
        'total_cost': total_cost,
        'rejection_cost_component': rejection_cost * rejection_rate,
        'error_cost_component': error_rate * coverage,
        'cost_without_rejection': 1 - accuracy_score(y_test, rejection_classifier.base_classifier.predict(X_test))
    }
    
    # Fairness analysis if groups provided
    fairness_metrics = {}
    if fairness_groups is not None:
        unique_groups = np.unique(groups_test)
        group_metrics = {}
        
        for group in unique_groups:
            group_mask = groups_test == group
            group_rejected = rejected_mask[group_mask]
            group_preds = predictions[group_mask]
            group_true = y_test[group_mask]
            
            group_rejection_rate = np.mean(group_rejected)
            group_non_rejected = ~group_rejected & (group_preds != -1)
            
            if np.any(group_non_rejected):
                group_accuracy = accuracy_score(group_true[group_non_rejected], group_preds[group_non_rejected])
            else:
                group_accuracy = 0.0
            
            group_metrics[f'group_{group}'] = {
                'rejection_rate': group_rejection_rate,
                'accuracy': group_accuracy,
                'coverage': 1 - group_rejection_rate,
                'size': np.sum(group_mask)
            }
        
        # Calculate fairness disparities
        rejection_rates = [metrics['rejection_rate'] for metrics in group_metrics.values()]
        accuracies = [metrics['accuracy'] for metrics in group_metrics.values()]
        
        fairness_metrics = {
            'group_metrics': group_metrics,
            'rejection_rate_disparity': np.max(rejection_rates) - np.min(rejection_rates),
            'accuracy_disparity': np.max(accuracies) - np.min(accuracies),
            'demographic_parity_rejection': np.std(rejection_rates),
            'equalized_odds_proxy': np.std(accuracies)