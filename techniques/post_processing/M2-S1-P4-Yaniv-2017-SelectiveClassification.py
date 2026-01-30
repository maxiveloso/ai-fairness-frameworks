import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, Callable
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
import warnings

def selective_classification(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    base_classifier: Optional[BaseEstimator] = None,
    selection_function: str = 'max_softmax',
    confidence_thresholds: Optional[Union[float, np.ndarray]] = None,
    test_size: float = 0.3,
    random_state: Optional[int] = None,
    custom_selection_func: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Implement selective classification with risk-coverage trade-off optimization.
    
    Selective classification allows a classifier to abstain from making predictions
    when confidence is low, trading coverage (fraction of examples classified) for
    accuracy. The algorithm trains a base classifier and uses a selection function
    to determine confidence, rejecting predictions below a threshold.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target labels
    base_classifier : BaseEstimator, optional
        Base classifier to use. If None, uses RandomForestClassifier
    selection_function : str, default='max_softmax'
        Selection function type: 'max_softmax', 'entropy', 'margin'
    confidence_thresholds : float or array-like, optional
        Confidence thresholds to evaluate. If None, uses range from 0.1 to 0.9
    test_size : float, default=0.3
        Proportion of dataset to use for testing
    random_state : int, optional
        Random state for reproducibility
    custom_selection_func : callable, optional
        Custom selection function that takes prediction probabilities and returns confidence scores
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'risk_coverage_curve': DataFrame with risk-coverage trade-off points
        - 'optimal_threshold': Threshold that minimizes selective risk
        - 'base_accuracy': Accuracy without selection
        - 'selective_metrics': Metrics at optimal threshold
        - 'auc_risk_coverage': Area under risk-coverage curve
        - 'selection_function': Selection function used
        - 'thresholds_evaluated': Thresholds that were tested
    """
    
    # Input validation
    X = np.asarray(X)
    y = np.asarray(y)
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
        
    if confidence_thresholds is not None:
        confidence_thresholds = np.asarray(confidence_thresholds)
        if np.any((confidence_thresholds < 0) | (confidence_thresholds > 1)):
            raise ValueError("confidence_thresholds must be between 0 and 1")
    
    # Set default confidence thresholds
    if confidence_thresholds is None:
        confidence_thresholds = np.linspace(0.1, 0.9, 17)
    else:
        confidence_thresholds = np.asarray(confidence_thresholds).flatten()
    
    # Set default base classifier
    if base_classifier is None:
        base_classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train base classifier
    base_classifier.fit(X_train, y_train)
    
    # Get prediction probabilities for test set
    if hasattr(base_classifier, 'predict_proba'):
        y_proba = base_classifier.predict_proba(X_test)
    else:
        raise ValueError("Base classifier must support predict_proba method")
    
    y_pred = base_classifier.predict(X_test)
    base_accuracy = accuracy_score(y_test, y_pred)
    
    # Define selection functions
    def max_softmax_confidence(proba):
        """Maximum softmax probability as confidence measure"""
        return np.max(proba, axis=1)
    
    def entropy_confidence(proba):
        """Negative entropy as confidence measure (higher = more confident)"""
        # Add small epsilon to avoid log(0)
        eps = 1e-15
        proba_clipped = np.clip(proba, eps, 1 - eps)
        entropy = -np.sum(proba_clipped * np.log(proba_clipped), axis=1)
        # Convert to confidence (higher = more confident)
        max_entropy = np.log(proba.shape[1])
        return 1 - (entropy / max_entropy)
    
    def margin_confidence(proba):
        """Margin between top two predictions as confidence measure"""
        sorted_proba = np.sort(proba, axis=1)
        return sorted_proba[:, -1] - sorted_proba[:, -2]
    
    # Select confidence function
    if custom_selection_func is not None:
        confidence_scores = custom_selection_func(y_proba)
        selection_func_name = 'custom'
    elif selection_function == 'max_softmax':
        confidence_scores = max_softmax_confidence(y_proba)
        selection_func_name = 'max_softmax'
    elif selection_function == 'entropy':
        confidence_scores = entropy_confidence(y_proba)
        selection_func_name = 'entropy'
    elif selection_function == 'margin':
        confidence_scores = margin_confidence(y_proba)
        selection_func_name = 'margin'
    else:
        raise ValueError("selection_function must be 'max_softmax', 'entropy', or 'margin'")
    
    # Evaluate risk-coverage trade-off for different thresholds
    risk_coverage_results = []
    
    for threshold in confidence_thresholds:
        # Select samples above confidence threshold
        selected_mask = confidence_scores >= threshold
        
        if np.sum(selected_mask) == 0:
            # No samples selected
            coverage = 0.0
            selective_risk = 1.0
            selective_accuracy = 0.0
            n_selected = 0
        else:
            # Calculate metrics for selected samples
            y_test_selected = y_test[selected_mask]
            y_pred_selected = y_pred[selected_mask]
            
            coverage = np.mean(selected_mask)  # Fraction of samples selected
            selective_accuracy = accuracy_score(y_test_selected, y_pred_selected)
            selective_risk = 1 - selective_accuracy  # Risk = 1 - Accuracy
            n_selected = np.sum(selected_mask)
        
        risk_coverage_results.append({
            'threshold': threshold,
            'coverage': coverage,
            'selective_risk': selective_risk,
            'selective_accuracy': selective_accuracy,
            'n_selected': n_selected
        })
    
    # Convert to DataFrame
    risk_coverage_df = pd.DataFrame(risk_coverage_results)
    
    # Find optimal threshold (minimize selective risk while maintaining reasonable coverage)
    # Use threshold that minimizes risk among those with coverage >= 0.1
    valid_thresholds = risk_coverage_df[risk_coverage_df['coverage'] >= 0.1]
    if len(valid_thresholds) > 0:
        optimal_idx = valid_thresholds['selective_risk'].idxmin()
        optimal_threshold = risk_coverage_df.loc[optimal_idx, 'threshold']
        optimal_metrics = risk_coverage_df.loc[optimal_idx].to_dict()
    else:
        # If no threshold gives coverage >= 0.1, use the one with highest coverage
        optimal_idx = risk_coverage_df['coverage'].idxmax()
        optimal_threshold = risk_coverage_df.loc[optimal_idx, 'threshold']
        optimal_metrics = risk_coverage_df.loc[optimal_idx].to_dict()
    
    # Calculate area under risk-coverage curve (lower is better)
    # Sort by coverage for proper integration
    sorted_df = risk_coverage_df.sort_values('coverage')
    if len(sorted_df) > 1:
        auc_risk_coverage = np.trapz(sorted_df['selective_risk'], sorted_df['coverage'])
    else:
        auc_risk_coverage = sorted_df['selective_risk'].iloc[0] if len(sorted_df) > 0 else 1.0
    
    # Get detailed metrics at optimal threshold
    optimal_mask = confidence_scores >= optimal_threshold
    if np.sum(optimal_mask) > 0:
        y_test_opt = y_test[optimal_mask]
        y_pred_opt = y_pred[optimal_mask]
        
        # Calculate additional metrics
        try:
            precision = precision_score(y_test_opt, y_pred_opt, average='weighted', zero_division=0)
            recall = recall_score(y_test_opt, y_pred_opt, average='weighted', zero_division=0)
            f1 = f1_score(y_test_opt, y_pred_opt, average='weighted', zero_division=0)
        except:
            precision = recall = f1 = 0.0
        
        detailed_metrics = {
            'accuracy': optimal_metrics['selective_accuracy'],
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'coverage': optimal_metrics['coverage'],
            'risk': optimal_metrics['selective_risk'],
            'n_selected': optimal_metrics['n_selected'],
            'n_total': len(y_test)
        }
    else:
        detailed_metrics = {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
            'coverage': 0.0, 'risk': 1.0, 'n_selected': 0, 'n_total': len(y_test)
        }
    
    return {
        'risk_coverage_curve': risk_coverage_df,
        'optimal_threshold': optimal_threshold,
        'base_accuracy': base_accuracy,
        'selective_metrics': detailed_metrics,
        'auc_risk_coverage': auc_risk_coverage,
        'selection_function': selection_func_name,
        'thresholds_evaluated': confidence_thresholds,
        'confidence_scores': confidence_scores,
        'base_classifier': base_classifier
    }


if __name__ == "__main__":
    # Example usage with synthetic dataset
    from sklearn.datasets import make_classification
    from sklearn.svm import SVC
    
    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    print("Selective Classification Example")
    print("=" * 40)
    
    # Example 1: Basic selective classification with default settings
    print("\n1. Basic Selective Classification:")
    results1 = selective_classification(X, y, random_state=42)
    
    print(f"Base classifier accuracy: {results1['base_accuracy']:.3f