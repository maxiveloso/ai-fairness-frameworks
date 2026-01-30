import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Union, Tuple, Dict, Any, Optional
import warnings

def platt_scaling(
    classifier_scores: Union[np.ndarray, pd.Series],
    true_labels: Union[np.ndarray, pd.Series],
    method: str = 'mle',
    cross_validation: bool = True,
    cv_folds: int = 3,
    regularization: float = 1e-7
) -> Dict[str, Any]:
    """
    Implement Platt Scaling to calibrate classifier outputs into probabilities.
    
    Platt Scaling fits a sigmoid function P(y=1|f) = 1/(1 + exp(Af + B)) to map
    classifier decision values f to calibrated probabilities. The parameters A and B
    are estimated via maximum likelihood estimation on a validation set to avoid
    overfitting.
    
    Parameters
    ----------
    classifier_scores : array-like of shape (n_samples,)
        Raw decision values or scores from a classifier (e.g., SVM decision function)
    true_labels : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    method : str, default='mle'
        Method for parameter estimation ('mle' for maximum likelihood or 'logistic')
    cross_validation : bool, default=True
        Whether to use cross-validation to avoid overfitting
    cv_folds : int, default=3
        Number of cross-validation folds if cross_validation=True
    regularization : float, default=1e-7
        Small regularization parameter to avoid numerical issues
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'A': Slope parameter of sigmoid function
        - 'B': Intercept parameter of sigmoid function  
        - 'calibrated_probabilities': Calibrated probabilities P(y=1|f)
        - 'log_likelihood': Log-likelihood of fitted model
        - 'original_scores': Original classifier scores
        - 'true_labels': True binary labels
        - 'method': Method used for calibration
        
    References
    ----------
    Platt, J. (1999). Probabilistic outputs for support vector machines and 
    comparisons to regularized likelihood methods. Advances in Large Margin 
    Classifiers, 10(3), 61-74.
    """
    
    # Input validation
    classifier_scores = np.asarray(classifier_scores).flatten()
    true_labels = np.asarray(true_labels).flatten()
    
    if len(classifier_scores) != len(true_labels):
        raise ValueError("classifier_scores and true_labels must have same length")
    
    if not np.all(np.isin(true_labels, [0, 1])):
        raise ValueError("true_labels must contain only 0 and 1")
    
    if len(np.unique(true_labels)) != 2:
        raise ValueError("true_labels must contain both classes (0 and 1)")
    
    if method not in ['mle', 'logistic']:
        raise ValueError("method must be 'mle' or 'logistic'")
    
    n_samples = len(classifier_scores)
    
    def sigmoid(scores: np.ndarray, A: float, B: float) -> np.ndarray:
        """Sigmoid function with numerical stability"""
        z = A * scores + B
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def negative_log_likelihood(params: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> float:
        """Negative log-likelihood for parameter optimization"""
        A, B = params
        probs = sigmoid(scores, A, B)
        # Add small epsilon to avoid log(0)
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)
        
        # Negative log-likelihood with L2 regularization
        nll = -np.sum(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        reg_term = regularization * (A**2 + B**2)
        return nll + reg_term
    
    def fit_platt_parameters(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """Fit Platt scaling parameters A and B"""
        if method == 'mle':
            # Maximum likelihood estimation using optimization
            initial_params = np.array([0.0, 0.0])
            
            result = minimize(
                negative_log_likelihood,
                initial_params,
                args=(scores, labels),
                method='BFGS',
                options={'maxiter': 1000}
            )
            
            if not result.success:
                warnings.warn("Optimization did not converge. Results may be unreliable.")
            
            A, B = result.x
            
        elif method == 'logistic':
            # Use logistic regression (equivalent to MLE for logistic model)
            lr = LogisticRegression(fit_intercept=True, C=1.0/regularization)
            lr.fit(scores.reshape(-1, 1), labels)
            A = lr.coef_[0][0]
            B = lr.intercept_[0]
        
        return A, B
    
    # Fit parameters with or without cross-validation
    if cross_validation and n_samples > cv_folds:
        # Use cross-validation to get unbiased probability estimates
        from sklearn.model_selection import StratifiedKFold
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        calibrated_probs = np.zeros(n_samples)
        A_values = []
        B_values = []
        
        for train_idx, val_idx in skf.split(classifier_scores, true_labels):
            train_scores = classifier_scores[train_idx]
            train_labels = true_labels[train_idx]
            val_scores = classifier_scores[val_idx]
            
            # Fit parameters on training fold
            A_fold, B_fold = fit_platt_parameters(train_scores, train_labels)
            A_values.append(A_fold)
            B_values.append(B_fold)
            
            # Apply to validation fold
            calibrated_probs[val_idx] = sigmoid(val_scores, A_fold, B_fold)
        
        # Average parameters across folds
        A = np.mean(A_values)
        B = np.mean(B_values)
        
    else:
        # Fit on entire dataset (may overfit)
        if cross_validation:
            warnings.warn("Not enough samples for cross-validation. Fitting on entire dataset.")
        
        A, B = fit_platt_parameters(classifier_scores, true_labels)
        calibrated_probs = sigmoid(classifier_scores, A, B)
    
    # Calculate final log-likelihood
    eps = 1e-15
    safe_probs = np.clip(calibrated_probs, eps, 1 - eps)
    log_likelihood = np.sum(
        true_labels * np.log(safe_probs) + 
        (1 - true_labels) * np.log(1 - safe_probs)
    )
    
    return {
        'A': A,
        'B': B,
        'calibrated_probabilities': calibrated_probs,
        'log_likelihood': log_likelihood,
        'original_scores': classifier_scores,
        'true_labels': true_labels,
        'method': method,
        'cross_validation': cross_validation,
        'n_samples': n_samples
    }

if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic classifier scores and true labels
    n_samples = 1000
    
    # Simulate SVM-like decision values
    true_labels = np.random.binomial(1, 0.4, n_samples)
    
    # Generate scores that are somewhat predictive but not calibrated
    classifier_scores = np.random.normal(
        loc=2 * true_labels - 1,  # Positive for class 1, negative for class 0
        scale=1.5,
        size=n_samples
    )
    
    print("Platt Scaling Example")
    print("=" * 50)
    print(f"Number of samples: {n_samples}")
    print(f"Class distribution: {np.mean(true_labels):.3f} positive class")
    print(f"Score range: [{np.min(classifier_scores):.3f}, {np.max(classifier_scores):.3f}]")
    
    # Apply Platt scaling with cross-validation
    results_cv = platt_scaling(
        classifier_scores=classifier_scores,
        true_labels=true_labels,
        method='mle',
        cross_validation=True,
        cv_folds=5
    )
    
    print(f"\nPlatt Scaling Results (with CV):")
    print(f"Parameter A (slope): {results_cv['A']:.4f}")
    print(f"Parameter B (intercept): {results_cv['B']:.4f}")
    print(f"Log-likelihood: {results_cv['log_likelihood']:.2f}")
    
    # Compare with logistic regression method
    results_lr = platt_scaling(
        classifier_scores=classifier_scores,
        true_labels=true_labels,
        method='logistic',
        cross_validation=True,
        cv_folds=5
    )
    
    print(f"\nLogistic Regression Method:")
    print(f"Parameter A (slope): {results_lr['A']:.4f}")
    print(f"Parameter B (intercept): {results_lr['B']:.4f}")
    print(f"Log-likelihood: {results_lr['log_likelihood']:.2f}")
    
    # Show calibration quality
    calibrated_probs = results_cv['calibrated_probabilities']
    print(f"\nCalibration Quality:")
    print(f"Probability range: [{np.min(calibrated_probs):.3f}, {np.max(calibrated_probs):.3f}]")
    print(f"Mean predicted probability: {np.mean(calibrated_probs):.3f}")
    print(f"True positive rate: {np.mean(true_labels):.3f}")
    
    # Reliability diagram (binned calibration assessment)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    print(f"\nReliability Diagram (Calibration Assessment):")
    print("Bin Range\t\tCount\tMean Pred\tTrue Freq\tDifference")
    print("-" * 65)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (calibrated_probs > bin_lower) & (calibrated_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = true_labels[in_bin].mean()
            avg_confidence_in_bin = calibrated_probs[in_bin].mean()
            count_in_bin = in_bin.sum()
            
            print(f"({bin_lower:.1f}, {bin_upper:.1f}]\t\t{