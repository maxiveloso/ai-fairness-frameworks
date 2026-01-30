import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, Any
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from scipy import stats

def reliability_diagrams(y_true: Union[np.ndarray, pd.Series], 
                        y_prob: Union[np.ndarray, pd.Series],
                        n_bins: int = 10,
                        strategy: str = 'uniform',
                        use_corp: bool = False,
                        return_plot_data: bool = True) -> Dict[str, Any]:
    """
    Generate reliability diagrams for probability calibration assessment.
    
    Reliability diagrams compare predicted probabilities against observed frequencies
    to assess how well a model's predicted probabilities reflect actual outcomes.
    Perfect calibration appears as points on the diagonal line (y=x).
    
    The CORP (Calibration with Optimal Reliability and Precision) method uses
    pool-adjacent-violators algorithm for optimal binning to reduce noise
    while preserving calibration assessment accuracy.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class
    n_bins : int, default=10
        Number of bins for grouping predicted probabilities
    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy for binning:
        - 'uniform': bins have equal width
        - 'quantile': bins have equal number of samples
    use_corp : bool, default=False
        Whether to use CORP method with pool-adjacent-violators algorithm
    return_plot_data : bool, default=True
        Whether to return data for plotting reliability diagram
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'fraction_of_positives': observed frequencies for each bin
        - 'mean_predicted_value': mean predicted probabilities for each bin
        - 'bin_counts': number of samples in each bin
        - 'reliability_score': reliability score (lower is better)
        - 'brier_score': Brier score for overall calibration
        - 'ece': Expected Calibration Error
        - 'mce': Maximum Calibration Error
        - 'calibration_slope': slope of calibration line
        - 'calibration_intercept': intercept of calibration line
        - 'plot_data': plotting coordinates (if return_plot_data=True)
        
    Raises
    ------
    ValueError
        If inputs have different lengths or invalid values
    """
    
    # Input validation
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only binary values (0, 1)")
    
    if not np.all((y_prob >= 0) & (y_prob <= 1)):
        raise ValueError("y_prob must contain probabilities between 0 and 1")
    
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    
    if strategy not in ['uniform', 'quantile']:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    
    if len(y_true) == 0:
        raise ValueError("No valid samples after removing NaN values")
    
    # Calculate calibration curve
    if use_corp:
        # Use isotonic regression (pool-adjacent-violators algorithm)
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        y_prob_calibrated = iso_reg.fit_transform(y_prob, y_true)
        
        # Create bins based on calibrated probabilities
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob_calibrated, n_bins=n_bins, strategy=strategy
        )
    else:
        # Standard binning approach
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy=strategy
        )
    
    # Calculate bin counts
    if strategy == 'uniform':
        bin_edges = np.linspace(0, 1, n_bins + 1)
    else:  # quantile
        bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = 0  # Ensure first edge is 0
        bin_edges[-1] = 1  # Ensure last edge is 1
    
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    
    # Filter out empty bins
    non_empty_bins = bin_counts > 0
    fraction_of_positives = fraction_of_positives[non_empty_bins[:len(fraction_of_positives)]]
    mean_predicted_value = mean_predicted_value[non_empty_bins[:len(mean_predicted_value)]]
    bin_counts = bin_counts[non_empty_bins]
    
    # Calculate reliability metrics
    
    # Reliability score (weighted mean squared difference from diagonal)
    reliability_score = np.average(
        (fraction_of_positives - mean_predicted_value) ** 2,
        weights=bin_counts
    )
    
    # Brier score
    brier_score = np.mean((y_prob - y_true) ** 2)
    
    # Expected Calibration Error (ECE)
    ece = np.average(
        np.abs(fraction_of_positives - mean_predicted_value),
        weights=bin_counts
    )
    
    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(fraction_of_positives - mean_predicted_value))
    
    # Calibration slope and intercept (linear regression)
    if len(mean_predicted_value) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            mean_predicted_value, fraction_of_positives
        )
        calibration_slope = slope
        calibration_intercept = intercept
    else:
        calibration_slope = np.nan
        calibration_intercept = np.nan
    
    # Prepare results
    results = {
        'fraction_of_positives': fraction_of_positives,
        'mean_predicted_value': mean_predicted_value,
        'bin_counts': bin_counts,
        'reliability_score': reliability_score,
        'brier_score': brier_score,
        'ece': ece,
        'mce': mce,
        'calibration_slope': calibration_slope,
        'calibration_intercept': calibration_intercept,
        'n_samples': len(y_true),
        'n_bins_used': len(fraction_of_positives)
    }
    
    # Add plot data if requested
    if return_plot_data:
        results['plot_data'] = {
            'x_perfect': np.linspace(0, 1, 100),
            'y_perfect': np.linspace(0, 1, 100),
            'x_observed': mean_predicted_value,
            'y_observed': fraction_of_positives,
            'bin_sizes': bin_counts
        }
    
    return results

if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic binary classification data
    n_samples = 1000
    
    # Create well-calibrated probabilities
    y_true_calibrated = np.random.binomial(1, 0.3, n_samples)
    y_prob_calibrated = np.random.beta(2, 5, n_samples)  # Well-calibrated
    
    # Create poorly-calibrated probabilities (overconfident)
    y_true_poor = np.random.binomial(1, 0.3, n_samples)
    y_prob_poor = np.random.beta(1, 2, n_samples)  # Overconfident
    y_prob_poor = np.clip(y_prob_poor * 1.5, 0, 1)  # Make more extreme
    
    print("Reliability Diagrams Analysis")
    print("=" * 50)
    
    # Analyze well-calibrated model
    print("\n1. Well-Calibrated Model:")
    results_good = reliability_diagrams(
        y_true_calibrated, y_prob_calibrated, 
        n_bins=10, strategy='uniform'
    )
    
    print(f"   Reliability Score: {results_good['reliability_score']:.4f}")
    print(f"   Brier Score: {results_good['brier_score']:.4f}")
    print(f"   Expected Calibration Error: {results_good['ece']:.4f}")
    print(f"   Maximum Calibration Error: {results_good['mce']:.4f}")
    print(f"   Calibration Slope: {results_good['calibration_slope']:.4f}")
    print(f"   Calibration Intercept: {results_good['calibration_intercept']:.4f}")
    
    # Analyze poorly-calibrated model
    print("\n2. Poorly-Calibrated Model:")
    results_poor = reliability_diagrams(
        y_true_poor, y_prob_poor, 
        n_bins=10, strategy='uniform'
    )
    
    print(f"   Reliability Score: {results_poor['reliability_score']:.4f}")
    print(f"   Brier Score: {results_poor['brier_score']:.4f}")
    print(f"   Expected Calibration Error: {results_poor['ece']:.4f}")
    print(f"   Maximum Calibration Error: {results_poor['mce']:.4f}")
    print(f"   Calibration Slope: {results_poor['calibration_slope']:.4f}")
    print(f"   Calibration Intercept: {results_poor['calibration_intercept']:.4f}")
    
    # Compare with CORP method
    print("\n3. CORP Method (Pool-Adjacent-Violators):")
    results_corp = reliability_diagrams(
        y_true_poor, y_prob_poor, 
        n_bins=10, strategy='uniform', use_corp=True
    )
    
    print(f"   Reliability Score: {results_corp['reliability_score']:.4f}")
    print(f"   Expected Calibration Error: {results_corp['ece']:.4f}")
    
    # Display bin information
    print(f"\n4. Binning Information:")
    print(f"   Number of bins used: {results_good['n_bins_used']}")
    print(f"   Bin counts: {results_good['bin_counts']}")
    print(f"   Mean predicted values: {results_good['mean_predicted_value']}")
    print(f"   Observed frequencies: {results_good['fraction_of_positives']}")
    
    print(f"\nInterpretation:")
    print(f"- Reliability Score closer to 0 indicates better calib