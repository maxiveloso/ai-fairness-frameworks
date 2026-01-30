import numpy as np
from scipy import optimize
from scipy.stats import beta
from typing import Union, Literal, Dict, Any
import warnings

def beta_calibration(
    predictions: Union[np.ndarray, list],
    labels: Union[np.ndarray, list],
    method: Literal['abm', 'ab', 'am'] = 'abm',
    epsilon: float = 1e-15
) -> Dict[str, Any]:
    """
    Perform Beta Calibration on probability predictions.
    
    Beta calibration models the relationship between predictions and outcomes using
    a beta distribution. Unlike Platt scaling which assumes Gaussian distributions,
    beta calibration assumes score distributions follow a ratio of two beta distributions.
    This method includes an identity function component and can be more flexible than
    logistic calibration for certain prediction distributions.
    
    The three fitting variants are:
    - 'abm': Fits parameters a, b, and m (scaling factor)
    - 'ab': Fits parameters a and b only (m fixed at 1)
    - 'am': Fits parameters a and m only (b derived from constraint)
    
    Parameters
    ----------
    predictions : array-like of shape (n_samples,)
        Probability predictions to be calibrated, should be in [0, 1]
    labels : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    method : {'abm', 'ab', 'am'}, default='abm'
        Fitting variant to use
    epsilon : float, default=1e-15
        Small value to avoid numerical issues at boundaries
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'parameters': Fitted parameters (a, b, m depending on method)
        - 'calibrated_predictions': Calibrated probability predictions
        - 'log_likelihood': Log-likelihood of the fitted model
        - 'method': Method used for fitting
        - 'calibration_function': Function to calibrate new predictions
        
    Raises
    ------
    ValueError
        If inputs are invalid or method is not recognized
    """
    # Input validation
    predictions = np.asarray(predictions, dtype=float)
    labels = np.asarray(labels, dtype=int)
    
    if predictions.shape[0] != labels.shape[0]:
        raise ValueError("predictions and labels must have the same length")
    
    if not np.all((predictions >= 0) & (predictions <= 1)):
        raise ValueError("predictions must be in [0, 1]")
    
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be binary (0 or 1)")
    
    if method not in ['abm', 'ab', 'am']:
        raise ValueError("method must be one of 'abm', 'ab', 'am'")
    
    # Clip predictions to avoid numerical issues
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    n_samples = len(predictions)
    n_positive = np.sum(labels)
    
    def beta_calibration_function(p, a, b, m=1.0):
        """
        Beta calibration transformation function.
        
        The calibrated probability is computed as:
        calibrated_p = (m * p^a) / (m * p^a + (1-p)^b)
        
        When m=1, this reduces to the standard beta calibration.
        """
        p_clipped = np.clip(p, epsilon, 1 - epsilon)
        numerator = m * (p_clipped ** a)
        denominator = numerator + ((1 - p_clipped) ** b)
        return numerator / denominator
    
    def negative_log_likelihood(params):
        """Compute negative log-likelihood for optimization."""
        if method == 'abm':
            a, b, m = params
            if a <= 0 or b <= 0 or m <= 0:
                return np.inf
        elif method == 'ab':
            a, b = params
            m = 1.0
            if a <= 0 or b <= 0:
                return np.inf
        elif method == 'am':
            a, m = params
            # For 'am' method, b is derived from the constraint that
            # the mean calibrated prediction equals the empirical frequency
            if a <= 0 or m <= 0:
                return np.inf
            # Use a reasonable default for b when using 'am' method
            b = a * (n_samples - n_positive) / n_positive if n_positive > 0 else 1.0
            if b <= 0:
                b = 1.0
        
        try:
            calibrated_probs = beta_calibration_function(predictions, a, b, m)
            calibrated_probs = np.clip(calibrated_probs, epsilon, 1 - epsilon)
            
            # Compute log-likelihood
            log_likelihood = np.sum(
                labels * np.log(calibrated_probs) + 
                (1 - labels) * np.log(1 - calibrated_probs)
            )
            return -log_likelihood
            
        except (ValueError, RuntimeWarning):
            return np.inf
    
    # Initialize parameters based on method
    if method == 'abm':
        # Initialize with reasonable starting values
        initial_params = [1.0, 1.0, 1.0]
        bounds = [(epsilon, 10), (epsilon, 10), (epsilon, 10)]
    elif method == 'ab':
        initial_params = [1.0, 1.0]
        bounds = [(epsilon, 10), (epsilon, 10)]
    elif method == 'am':
        initial_params = [1.0, 1.0]
        bounds = [(epsilon, 10), (epsilon, 10)]
    
    # Optimize parameters
    try:
        result = optimize.minimize(
            negative_log_likelihood,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        if not result.success:
            warnings.warn("Optimization did not converge successfully")
        
        optimal_params = result.x
        
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}")
        optimal_params = initial_params
    
    # Extract fitted parameters
    if method == 'abm':
        a_fit, b_fit, m_fit = optimal_params
        fitted_params = {'a': a_fit, 'b': b_fit, 'm': m_fit}
    elif method == 'ab':
        a_fit, b_fit = optimal_params
        m_fit = 1.0
        fitted_params = {'a': a_fit, 'b': b_fit, 'm': m_fit}
    elif method == 'am':
        a_fit, m_fit = optimal_params
        b_fit = a_fit * (n_samples - n_positive) / n_positive if n_positive > 0 else 1.0
        if b_fit <= 0:
            b_fit = 1.0
        fitted_params = {'a': a_fit, 'b': b_fit, 'm': m_fit}
    
    # Generate calibrated predictions
    calibrated_preds = beta_calibration_function(predictions, a_fit, b_fit, m_fit)
    
    # Compute final log-likelihood
    calibrated_preds_clipped = np.clip(calibrated_preds, epsilon, 1 - epsilon)
    final_log_likelihood = np.sum(
        labels * np.log(calibrated_preds_clipped) + 
        (1 - labels) * np.log(1 - calibrated_preds_clipped)
    )
    
    # Create calibration function for new data
    def calibration_function(new_predictions):
        """Apply fitted calibration to new predictions."""
        new_predictions = np.asarray(new_predictions)
        return beta_calibration_function(new_predictions, a_fit, b_fit, m_fit)
    
    return {
        'parameters': fitted_params,
        'calibrated_predictions': calibrated_preds,
        'log_likelihood': final_log_likelihood,
        'method': method,
        'calibration_function': calibration_function
    }

if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic probability predictions and labels
    n_samples = 1000
    true_probs = np.random.beta(2, 3, n_samples)  # True probabilities from beta distribution
    
    # Add miscalibration - predictions are systematically overconfident
    predictions = np.clip(true_probs ** 0.7, 0.01, 0.99)
    
    # Generate binary labels based on true probabilities
    labels = np.random.binomial(1, true_probs, n_samples)
    
    print("Beta Calibration Example")
    print("=" * 50)
    print(f"Number of samples: {n_samples}")
    print(f"Positive class rate: {np.mean(labels):.3f}")
    print(f"Mean prediction: {np.mean(predictions):.3f}")
    print()
    
    # Test different methods
    methods = ['abm', 'ab', 'am']
    
    for method in methods:
        print(f"Method: {method}")
        print("-" * 20)
        
        result = beta_calibration(predictions, labels, method=method)
        
        print(f"Fitted parameters: {result['parameters']}")
        print(f"Log-likelihood: {result['log_likelihood']:.2f}")
        print(f"Mean calibrated prediction: {np.mean(result['calibrated_predictions']):.3f}")
        
        # Test calibration function on new data
        test_preds = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        calibrated_test = result['calibration_function'](test_preds)
        print(f"Test predictions: {test_preds}")
        print(f"Calibrated: {calibrated_test}")
        print()