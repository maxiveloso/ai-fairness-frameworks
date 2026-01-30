import numpy as np
from typing import Union, Dict, List
import warnings

def expected_calibration_error(y_true: Union[np.ndarray, List], 
                             y_prob: Union[np.ndarray, List], 
                             n_bins: int = 10,
                             strategy: str = 'uniform') -> Dict[str, Union[float, np.ndarray, List]]:
    """
    Calculate Expected Calibration Error (ECE) to measure model calibration.
    
    ECE measures the difference between predicted confidence and actual accuracy
    across different confidence levels. A well-calibrated model should have
    predicted probabilities that match the true frequencies of positive outcomes.
    
    The ECE is computed as:
    ECE = Σᵢᴺ bᵢ ||(pᵢ - cᵢ)||
    
    where:
    - pᵢ is the accuracy (fraction of correct predictions) in bin i
    - cᵢ is the average confidence (mean predicted probability) in bin i  
    - bᵢ is the fraction of total samples in bin i
    - N is the number of bins
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class, must be in [0, 1]
    n_bins : int, default=10
        Number of bins to divide the probability space [0, 1]
    strategy : str, default='uniform'
        Strategy for creating bins. Currently only 'uniform' is supported,
        which creates equal-width bins across [0, 1]
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'ece': float, the Expected Calibration Error
        - 'bin_boundaries': array, the boundaries of the bins used
        - 'bin_accuracies': array, accuracy within each bin
        - 'bin_confidences': array, average confidence within each bin
        - 'bin_counts': array, number of samples in each bin
        - 'bin_fractions': array, fraction of total samples in each bin
        - 'calibration_gaps': array, absolute difference between accuracy and confidence per bin
    
    Raises
    ------
    ValueError
        If inputs have different lengths, contain invalid values, or parameters are invalid
    
    Notes
    -----
    - ECE ranges from 0 to 1, where 0 indicates perfect calibration
    - Lower ECE values indicate better calibration
    - Empty bins contribute 0 to the ECE calculation
    - This implementation assumes binary classification problems
    
    References
    ----------
    Hauskrecht (2015). Expected Calibration Error for model calibration assessment.
    
    Examples
    --------
    >>> y_true = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
    >>> y_prob = [0.1, 0.3, 0.6, 0.8, 0.9, 0.2, 0.7, 0.4, 0.85, 0.95]
    >>> result = expected_calibration_error(y_true, y_prob, n_bins=5)
    >>> print(f"ECE: {result['ece']:.4f}")
    """
    
    # Input validation
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    
    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only binary values (0 or 1)")
    
    if not np.all((y_prob >= 0) & (y_prob <= 1)):
        raise ValueError("y_prob must contain probabilities in range [0, 1]")
    
    if not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    
    if strategy != 'uniform':
        raise ValueError("Only 'uniform' binning strategy is currently supported")
    
    # Create bin boundaries for uniform strategy
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    # Initialize arrays to store bin statistics
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    bin_fractions = np.zeros(n_bins)
    calibration_gaps = np.zeros(n_bins)
    
    # Assign each prediction to a bin
    # Use digitize to find which bin each probability belongs to
    bin_indices = np.digitize(y_prob, bin_boundaries, right=False)
    # Handle edge case where probability = 1.0 (assign to last bin)
    bin_indices = np.clip(bin_indices - 1, 0, n_bins - 1)
    
    total_samples = len(y_true)
    
    # Calculate statistics for each bin
    for i in range(n_bins):
        # Find samples in current bin
        mask = (bin_indices == i)
        bin_counts[i] = np.sum(mask)
        
        if bin_counts[i] > 0:
            # Calculate accuracy (fraction of correct predictions in bin)
            bin_true_labels = y_true[mask]
            bin_pred_probs = y_prob[mask]
            
            # For binary classification, accuracy is the fraction of positive labels
            bin_accuracies[i] = np.mean(bin_true_labels)
            
            # Average confidence is mean predicted probability in bin
            bin_confidences[i] = np.mean(bin_pred_probs)
            
            # Fraction of total samples in this bin
            bin_fractions[i] = bin_counts[i] / total_samples
            
            # Calibration gap (absolute difference between accuracy and confidence)
            calibration_gaps[i] = abs(bin_accuracies[i] - bin_confidences[i])
        else:
            # Empty bin contributes 0 to ECE
            bin_accuracies[i] = 0
            bin_confidences[i] = 0
            bin_fractions[i] = 0
            calibration_gaps[i] = 0
    
    # Calculate Expected Calibration Error
    # ECE = Σᵢ bᵢ * |pᵢ - cᵢ| where bᵢ is bin fraction, pᵢ is accuracy, cᵢ is confidence
    ece = np.sum(bin_fractions * calibration_gaps)
    
    # Prepare results dictionary
    results = {
        'ece': float(ece),
        'bin_boundaries': bin_boundaries,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
        'bin_fractions': bin_fractions,
        'calibration_gaps': calibration_gaps
    }
    
    return results


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic binary classification data
    n_samples = 1000
    
    # Create true labels
    y_true = np.random.binomial(1, 0.4, n_samples)
    
    # Create predicted probabilities with some miscalibration
    # Well-calibrated probabilities
    y_prob_calibrated = np.random.beta(2, 3, n_samples)
    
    # Overconfident probabilities (shifted towards extremes)
    y_prob_overconfident = np.where(y_prob_calibrated > 0.5, 
                                   y_prob_calibrated * 1.3, 
                                   y_prob_calibrated * 0.7)
    y_prob_overconfident = np.clip(y_prob_overconfident, 0, 1)
    
    print("Expected Calibration Error (ECE) Analysis")
    print("=" * 50)
    
    # Test with well-calibrated probabilities
    print("\n1. Well-calibrated model:")
    result_calibrated = expected_calibration_error(y_true, y_prob_calibrated, n_bins=10)
    print(f"   ECE: {result_calibrated['ece']:.4f}")
    print(f"   Number of bins: {len(result_calibrated['bin_boundaries']) - 1}")
    print(f"   Average calibration gap: {np.mean(result_calibrated['calibration_gaps']):.4f}")
    
    # Test with overconfident probabilities
    print("\n2. Overconfident model:")
    result_overconfident = expected_calibration_error(y_true, y_prob_overconfident, n_bins=10)
    print(f"   ECE: {result_overconfident['ece']:.4f}")
    print(f"   Number of bins: {len(result_overconfident['bin_boundaries']) - 1}")
    print(f"   Average calibration gap: {np.mean(result_overconfident['calibration_gaps']):.4f}")
    
    # Show detailed bin analysis for overconfident model
    print("\n3. Detailed bin analysis (overconfident model):")
    print("   Bin Range        | Count | Accuracy | Confidence | Gap")
    print("   " + "-" * 55)
    
    for i in range(len(result_overconfident['bin_counts'])):
        if result_overconfident['bin_counts'][i] > 0:
            bin_start = result_overconfident['bin_boundaries'][i]
            bin_end = result_overconfident['bin_boundaries'][i + 1]
            count = result_overconfident['bin_counts'][i]
            accuracy = result_overconfident['bin_accuracies'][i]
            confidence = result_overconfident['bin_confidences'][i]
            gap = result_overconfident['calibration_gaps'][i]
            
            print(f"   [{bin_start:.1f}, {bin_end:.1f}) | {count:5d} | {accuracy:8.3f} | {confidence:10.3f} | {gap:.3f}")
    
    # Test with different number of bins
    print("\n4. Effect of different bin sizes:")
    for n_bins in [5, 10, 15, 20]:
        result = expected_calibration_error(y_true, y_prob_overconfident, n_bins=n_bins)
        print(f"   {n_bins:2d} bins: ECE = {result['ece']:.4f}")
    
    # Test edge cases
    print("\n5. Edge case testing:")
    
    # Perfect calibration
    y_perfect = np.array([0, 0, 1, 1])
    prob_perfect = np.array([0.0, 0.0, 1.0, 1.0])
    result_perfect = expected_calibration_error(y_perfect, prob_perfect, n_bins=4)
    print(f"   Perfect calibration ECE: {result_perfect['ece']:.4f}")
    
    # Worst calibration
    y_worst = np.array([0, 0, 1, 1])
    prob_worst = np.array([1.0, 1.0, 0.0, 0.0])
    result_worst = expected_calibration_error(y_worst, prob_worst, n_bins=4)
    print(f"   Worst calibration ECE: {result_worst['ece']:.4f}")