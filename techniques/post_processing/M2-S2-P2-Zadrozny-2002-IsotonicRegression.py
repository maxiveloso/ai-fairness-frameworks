import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, brier_score_loss
import warnings

def isotonic_regression(
    y_true: Union[np.ndarray, list],
    y_scores: Union[np.ndarray, list],
    sample_weight: Optional[Union[np.ndarray, list]] = None,
    increasing: bool = True,
    out_of_bounds: str = 'nan'
) -> Dict[str, Any]:
    """
    Perform isotonic regression for probability calibration using Pool Adjacent Violators algorithm.
    
    Isotonic regression finds a non-decreasing approximation of a function while 
    minimizing the mean squared error. It's commonly used to calibrate classifier
    output scores into well-calibrated probabilities by enforcing monotonicity
    constraints. The Pool Adjacent Violators (PAV) algorithm efficiently solves
    this optimization problem in O(n) time complexity.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1) or continuous target values
    y_scores : array-like of shape (n_samples,)
        Raw classifier scores or uncalibrated probabilities to be calibrated
    sample_weight : array-like of shape (n_samples,), optional
        Individual weights for each sample. If None, uniform weights are used
    increasing : bool, default=True
        Whether the isotonic function should be increasing (True) or decreasing (False)
    out_of_bounds : {'nan', 'clip', 'raise'}, default='nan'
        How to handle out-of-bounds values during prediction
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'calibrated_probabilities': Calibrated probability estimates
        - 'isotonic_function_x': X coordinates of the isotonic function
        - 'isotonic_function_y': Y coordinates of the isotonic function  
        - 'mse_before': Mean squared error before calibration
        - 'mse_after': Mean squared error after calibration
        - 'brier_score_before': Brier score before calibration (if binary)
        - 'brier_score_after': Brier score after calibration (if binary)
        - 'calibration_improvement': Relative improvement in MSE
        - 'model': Fitted IsotonicRegression model for future predictions
        
    Raises
    ------
    ValueError
        If input arrays have different lengths or contain invalid values
    """
    
    # Input validation
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length")
    
    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_scores)):
        raise ValueError("Input arrays cannot contain NaN values")
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if len(sample_weight) != len(y_true):
            raise ValueError("sample_weight must have the same length as y_true")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values")
    
    if out_of_bounds not in ['nan', 'clip', 'raise']:
        raise ValueError("out_of_bounds must be one of 'nan', 'clip', 'raise'")
    
    # Check if binary classification problem
    unique_labels = np.unique(y_true)
    is_binary = len(unique_labels) == 2 and set(unique_labels).issubset({0, 1})
    
    # Calculate metrics before calibration
    mse_before = mean_squared_error(y_true, y_scores, sample_weight=sample_weight)
    brier_before = None
    if is_binary:
        # Clip scores to [0,1] for Brier score calculation
        y_scores_clipped = np.clip(y_scores, 0, 1)
        brier_before = brier_score_loss(y_true, y_scores_clipped, sample_weight=sample_weight)
    
    # Fit isotonic regression model
    # The PAV algorithm works by pooling adjacent violators of the monotonicity constraint
    iso_reg = IsotonicRegression(
        increasing=increasing,
        out_of_bounds=out_of_bounds
    )
    
    # Fit the model and get calibrated probabilities
    calibrated_probs = iso_reg.fit_transform(y_scores, y_true, sample_weight=sample_weight)
    
    # Calculate metrics after calibration
    mse_after = mean_squared_error(y_true, calibrated_probs, sample_weight=sample_weight)
    brier_after = None
    if is_binary:
        brier_after = brier_score_loss(y_true, calibrated_probs, sample_weight=sample_weight)
    
    # Calculate calibration improvement
    calibration_improvement = (mse_before - mse_after) / mse_before if mse_before > 0 else 0.0
    
    # Extract the isotonic function coordinates
    # These represent the piecewise-constant monotonic function learned by PAV
    function_x = iso_reg.X_thresholds_
    function_y = iso_reg.y_thresholds_
    
    # Prepare results dictionary
    results = {
        'calibrated_probabilities': calibrated_probs,
        'isotonic_function_x': function_x,
        'isotonic_function_y': function_y,
        'mse_before': mse_before,
        'mse_after': mse_after,
        'calibration_improvement': calibration_improvement,
        'model': iso_reg,
        'n_samples': len(y_true),
        'n_thresholds': len(function_x)
    }
    
    # Add binary classification specific metrics
    if is_binary:
        results['brier_score_before'] = brier_before
        results['brier_score_after'] = brier_after
        results['brier_improvement'] = (brier_before - brier_after) / brier_before if brier_before > 0 else 0.0
    
    return results

if __name__ == "__main__":
    # Example 1: Binary classification calibration
    print("Example 1: Binary Classification Probability Calibration")
    print("=" * 60)
    
    # Simulate poorly calibrated classifier scores
    np.random.seed(42)
    n_samples = 1000
    
    # Generate true labels
    y_true_binary = np.random.binomial(1, 0.3, n_samples)
    
    # Generate poorly calibrated scores (overconfident)
    y_scores_binary = np.random.beta(2, 5, n_samples)
    y_scores_binary[y_true_binary == 1] += 0.4
    y_scores_binary = np.clip(y_scores_binary, 0, 1)
    
    # Apply isotonic regression calibration
    results_binary = isotonic_regression(y_true_binary, y_scores_binary)
    
    print(f"Original MSE: {results_binary['mse_before']:.4f}")
    print(f"Calibrated MSE: {results_binary['mse_after']:.4f}")
    print(f"MSE Improvement: {results_binary['calibration_improvement']:.2%}")
    print(f"Original Brier Score: {results_binary['brier_score_before']:.4f}")
    print(f"Calibrated Brier Score: {results_binary['brier_score_after']:.4f}")
    print(f"Brier Score Improvement: {results_binary['brier_improvement']:.2%}")
    print(f"Number of isotonic function thresholds: {results_binary['n_thresholds']}")
    
    # Example 2: Regression with monotonicity constraint
    print("\n\nExample 2: Regression with Monotonicity Constraint")
    print("=" * 60)
    
    # Generate noisy monotonic data
    np.random.seed(123)
    x_reg = np.linspace(0, 10, 200)
    y_true_reg = 2 * np.log(x_reg + 1) + np.random.normal(0, 0.5, len(x_reg))
    y_scores_reg = y_true_reg + np.random.normal(0, 1, len(x_reg))  # Add more noise
    
    # Apply isotonic regression
    results_reg = isotonic_regression(y_true_reg, y_scores_reg)
    
    print(f"Original MSE: {results_reg['mse_before']:.4f}")
    print(f"Isotonic MSE: {results_reg['mse_after']:.4f}")
    print(f"MSE Improvement: {results_reg['calibration_improvement']:.2%}")
    print(f"Number of samples: {results_reg['n_samples']}")
    print(f"Number of isotonic function segments: {results_reg['n_thresholds']}")
    
    # Example 3: Using sample weights
    print("\n\nExample 3: Weighted Isotonic Regression")
    print("=" * 60)
    
    # Create sample weights (higher weights for more reliable samples)
    sample_weights = np.random.uniform(0.5, 2.0, len(y_true_binary))
    
    results_weighted = isotonic_regression(
        y_true_binary, 
        y_scores_binary, 
        sample_weight=sample_weights
    )
    
    print(f"Weighted MSE before: {results_weighted['mse_before']:.4f}")
    print(f"Weighted MSE after: {results_weighted['mse_after']:.4f}")
    print(f"Weighted improvement: {results_weighted['calibration_improvement']:.2%}")
    
    # Demonstrate prediction on new data
    print("\n\nExample 4: Prediction on New Data")
    print("=" * 60)
    
    # Generate new test scores
    new_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    calibrated_model = results_binary['model']
    new_calibrated = calibrated_model.predict(new_scores)
    
    print("Original scores -> Calibrated probabilities:")
    for orig, calib in zip(new_scores, new_calibrated):
        print(f"  {orig:.1f} -> {calib:.3f}")