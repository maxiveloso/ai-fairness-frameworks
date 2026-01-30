import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, Tuple
from scipy.optimize import minimize
from sklearn.base import BaseClassifier, clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

def fairness_constraints_classification(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    sensitive_attr: Union[np.ndarray, pd.Series],
    classifier: Optional[BaseClassifier] = None,
    fairness_type: str = 'disparate_impact',
    fairness_constraint: float = 0.05,
    c_fairness: float = 1.0,
    max_iter: int = 1000,
    tolerance: float = 1e-6,
    standardize: bool = True
) -> Dict[str, Any]:
    """
    Implement fairness constraints for fair classification using decision boundary unfairness.
    
    This technique implements fairness constraints as proposed by Zafar et al. (2017) that
    can be incorporated into convex loss functions. The method uses a tractable proxy for
    unfairness based on the decision boundary and supports both disparate impact and
    disparate mistreatment constraints.
    
    The key insight is to add fairness constraints directly to the optimization objective:
    - Disparate Impact: Ensures similar positive prediction rates across groups
    - Disparate Mistreatment: Ensures similar error rates across groups
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Binary target variable (0 or 1)
    sensitive_attr : array-like of shape (n_samples,)
        Binary sensitive attribute (0 or 1, e.g., gender, race)
    classifier : BaseClassifier, optional
        Base classifier to use. If None, uses LogisticRegression
    fairness_type : str, default='disparate_impact'
        Type of fairness constraint ('disparate_impact' or 'disparate_mistreatment')
    fairness_constraint : float, default=0.05
        Maximum allowed unfairness (smaller values = more fair)
    c_fairness : float, default=1.0
        Weight of fairness constraint relative to accuracy
    max_iter : int, default=1000
        Maximum number of optimization iterations
    tolerance : float, default=1e-6
        Convergence tolerance for optimization
    standardize : bool, default=True
        Whether to standardize features
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'fair_classifier': Trained fair classifier
        - 'unfair_classifier': Baseline unfair classifier
        - 'fair_accuracy': Accuracy of fair classifier
        - 'unfair_accuracy': Accuracy of unfair classifier
        - 'fairness_violation_fair': Fairness violation of fair classifier
        - 'fairness_violation_unfair': Fairness violation of unfair classifier
        - 'disparate_impact_fair': Disparate impact ratio for fair classifier
        - 'disparate_impact_unfair': Disparate impact ratio for unfair classifier
        - 'group_statistics': Detailed statistics by group
        - 'optimization_info': Information about the optimization process
        
    References
    ----------
    Zafar, M. B., Valera, I., Gomez Rodriguez, M., & Gummadi, K. P. (2017).
    Fairness constraints: Mechanisms for fair classification. In Artificial
    Intelligence and Statistics (pp. 962-970).
    """
    
    # Input validation
    X = np.asarray(X)
    y = np.asarray(y)
    sensitive_attr = np.asarray(sensitive_attr)
    
    if X.shape[0] != len(y) or X.shape[0] != len(sensitive_attr):
        raise ValueError("X, y, and sensitive_attr must have the same number of samples")
    
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must be binary (0 or 1)")
    
    if not np.all(np.isin(sensitive_attr, [0, 1])):
        raise ValueError("sensitive_attr must be binary (0 or 1)")
    
    if fairness_type not in ['disparate_impact', 'disparate_mistreatment']:
        raise ValueError("fairness_type must be 'disparate_impact' or 'disparate_mistreatment'")
    
    if not 0 <= fairness_constraint <= 1:
        raise ValueError("fairness_constraint must be between 0 and 1")
    
    n_samples, n_features = X.shape
    
    # Standardize features if requested
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.copy()
        scaler = None
    
    # Set default classifier
    if classifier is None:
        classifier = LogisticRegression(random_state=42)
    
    # Train baseline unfair classifier
    unfair_classifier = clone(classifier)
    unfair_classifier.fit(X_scaled, y)
    unfair_predictions = unfair_classifier.predict(X_scaled)
    unfair_accuracy = accuracy_score(y, unfair_predictions)
    
    # Calculate fairness violations for unfair classifier
    unfair_fairness_violation = _calculate_fairness_violation(
        y, unfair_predictions, sensitive_attr, fairness_type
    )
    unfair_disparate_impact = _calculate_disparate_impact(
        unfair_predictions, sensitive_attr
    )
    
    # Implement fair classifier using constrained optimization
    fair_classifier = _FairClassifier(
        base_classifier=classifier,
        fairness_type=fairness_type,
        fairness_constraint=fairness_constraint,
        c_fairness=c_fairness,
        max_iter=max_iter,
        tolerance=tolerance
    )
    
    # Fit fair classifier
    optimization_info = fair_classifier.fit(X_scaled, y, sensitive_attr)
    fair_predictions = fair_classifier.predict(X_scaled)
    fair_accuracy = accuracy_score(y, fair_predictions)
    
    # Calculate fairness violations for fair classifier
    fair_fairness_violation = _calculate_fairness_violation(
        y, fair_predictions, sensitive_attr, fairness_type
    )
    fair_disparate_impact = _calculate_disparate_impact(
        fair_predictions, sensitive_attr
    )
    
    # Calculate detailed group statistics
    group_stats = _calculate_group_statistics(
        y, fair_predictions, unfair_predictions, sensitive_attr
    )
    
    return {
        'fair_classifier': fair_classifier,
        'unfair_classifier': unfair_classifier,
        'fair_accuracy': fair_accuracy,
        'unfair_accuracy': unfair_accuracy,
        'fairness_violation_fair': fair_fairness_violation,
        'fairness_violation_unfair': unfair_fairness_violation,
        'disparate_impact_fair': fair_disparate_impact,
        'disparate_impact_unfair': unfair_disparate_impact,
        'group_statistics': group_stats,
        'optimization_info': optimization_info,
        'scaler': scaler
    }


class _FairClassifier:
    """
    Internal fair classifier implementation using constrained optimization.
    
    This class implements the core fairness-constrained optimization problem
    by adding fairness constraints to the loss function of the base classifier.
    """
    
    def __init__(self, base_classifier, fairness_type, fairness_constraint, 
                 c_fairness, max_iter, tolerance):
        self.base_classifier = base_classifier
        self.fairness_type = fairness_type
        self.fairness_constraint = fairness_constraint
        self.c_fairness = c_fairness
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y, sensitive_attr):
        """Fit the fair classifier using constrained optimization."""
        n_samples, n_features = X.shape
        
        # Initialize parameters from base classifier
        base_clf = clone(self.base_classifier)
        base_clf.fit(X, y)
        
        if hasattr(base_clf, 'coef_') and hasattr(base_clf, 'intercept_'):
            # For linear models like LogisticRegression
            initial_params = np.concatenate([
                base_clf.coef_.flatten(), 
                base_clf.intercept_.flatten()
            ])
        else:
            # For non-linear models, use random initialization
            initial_params = np.random.normal(0, 0.1, n_features + 1)
        
        # Define objective function
        def objective(params):
            coef = params[:-1]
            intercept = params[-1]
            
            # Calculate predictions (logistic function)
            scores = X @ coef + intercept
            probs = 1 / (1 + np.exp(-np.clip(scores, -500, 500)))  # Numerical stability
            
            # Logistic loss
            epsilon = 1e-15  # Prevent log(0)
            probs = np.clip(probs, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
            
            # Fairness constraint penalty
            predictions = (probs > 0.5).astype(int)
            fairness_violation = _calculate_fairness_violation(
                y, predictions, sensitive_attr, self.fairness_type
            )
            
            # Add fairness penalty if constraint is violated
            fairness_penalty = 0
            if fairness_violation > self.fairness_constraint:
                fairness_penalty = self.c_fairness * (fairness_violation - self.fairness_constraint) ** 2
            
            return loss + fairness_penalty
        
        # Optimize
        try:
            result = minimize(
                objective,
                initial_params,
                method='BFGS',
                options={'maxiter': self.max_iter, 'gtol': self.tolerance}
            )
            
            self.coef_ = result.x[:-1]
            self.intercept_ = result.x[-1]
            
            optimization_info = {
                'success': result.success,
                'message': result.message,
                'n_iterations': result.nit,
                'final_loss': result.fun
            }
            
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}. Using base classifier.")
            self.coef_ = base_clf.coef_.flatten() if hasattr(base_clf, 'coef_') else np.random.normal(0, 0.1, n_features)
            self.intercept_ = base_clf.intercept_[0] if hasattr(base_clf, 'intercept_') else 0.0
            
            optimization_info = {
                'success': False,
                'message': str(e),
                'n_iterations': 0,
                'final_loss': np.inf
            }
        
        return optimization_info
    
    def predict(self, X):
        """Make predictions using the fitted fair classifier."""
        if self.coef_ is None:
            raise ValueError("Classifier must be fitted before making predictions")
        
        scores = X @ self.coef_ + self.intercept_
        probs = 1 / (1 + np.exp(-np.clip(scores, -500, 500)))
        return (probs > 0.5).astype(