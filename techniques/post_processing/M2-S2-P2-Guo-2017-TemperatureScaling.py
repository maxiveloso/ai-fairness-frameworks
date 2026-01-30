import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
from typing import Union, Dict, Optional, Tuple
import warnings

def temperature_scaling(
    logits: Union[np.ndarray, pd.DataFrame],
    true_labels: Union[np.ndarray, pd.Series],
    validation_split: float = 0.2,
    temperature_bounds: Tuple[float, float] = (0.1, 10.0),
    random_state: Optional[int] = None,
    group_column: Optional[str] = None
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Apply temperature scaling calibration to neural network logits.
    
    Temperature scaling is a post-processing calibration technique that applies
    a single scalar parameter T to all logits before the softmax operation.
    The temperature T is optimized to minimize negative log-likelihood on a
    validation set. This simple technique often significantly improves
    calibration of modern neural networks.
    
    The calibrated probability is computed as:
    P(y=i|x) = exp(z_i/T) / Σ_j exp(z_j/T)
    
    where z_i are the original logits and T is the learned temperature parameter.
    
    Parameters
    ----------
    logits : array-like of shape (n_samples, n_classes)
        Raw logits from neural network before softmax
    true_labels : array-like of shape (n_samples,)
        True class labels (integer encoded)
    validation_split : float, default=0.2
        Fraction of data to use for temperature optimization
    temperature_bounds : tuple of float, default=(0.1, 10.0)
        Lower and upper bounds for temperature parameter search
    random_state : int, optional
        Random seed for train/validation split
    group_column : str, optional
        If provided and logits is DataFrame, apply separate temperature
        scaling for each group in this column
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'temperature': Optimal temperature parameter(s)
        - 'calibrated_probabilities': Softmax probabilities after temperature scaling
        - 'original_probabilities': Original softmax probabilities
        - 'validation_nll_before': NLL before calibration on validation set
        - 'validation_nll_after': NLL after calibration on validation set
        - 'group_temperatures': Dict of temperatures per group (if group_column provided)
    """
    
    # Input validation
    if isinstance(logits, pd.DataFrame):
        if group_column and group_column not in logits.columns:
            raise ValueError(f"Group column '{group_column}' not found in logits DataFrame")
        logits_array = logits.drop(columns=[group_column] if group_column else []).values
        groups = logits[group_column].values if group_column else None
    else:
        logits_array = np.asarray(logits)
        groups = None
        
    true_labels = np.asarray(true_labels)
    
    if logits_array.ndim != 2:
        raise ValueError("Logits must be 2-dimensional (n_samples, n_classes)")
    
    if len(logits_array) != len(true_labels):
        raise ValueError("Number of samples in logits and true_labels must match")
        
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1")
        
    if temperature_bounds[0] >= temperature_bounds[1]:
        raise ValueError("temperature_bounds[0] must be less than temperature_bounds[1]")
    
    n_samples, n_classes = logits_array.shape
    
    # Check if labels are valid
    unique_labels = np.unique(true_labels)
    if np.min(unique_labels) < 0 or np.max(unique_labels) >= n_classes:
        raise ValueError(f"Labels must be in range [0, {n_classes-1}]")
    
    def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
        """Apply softmax with temperature scaling"""
        scaled_logits = logits / temperature
        # Numerical stability: subtract max
        scaled_logits = scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)
        exp_logits = np.exp(scaled_logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
        """Compute negative log-likelihood"""
        # Clip probabilities to avoid log(0)
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        return -np.mean(np.log(probs[np.arange(len(labels)), labels]))
    
    def optimize_temperature(train_logits: np.ndarray, train_labels: np.ndarray,
                           val_logits: np.ndarray, val_labels: np.ndarray) -> float:
        """Find optimal temperature parameter"""
        def objective(temperature: float) -> float:
            val_probs = softmax_with_temperature(val_logits, temperature)
            return negative_log_likelihood(val_probs, val_labels)
        
        result = minimize_scalar(objective, bounds=temperature_bounds, method='bounded')
        if not result.success:
            warnings.warn("Temperature optimization did not converge")
        
        return result.x
    
    # Original probabilities (temperature = 1.0)
    original_probs = softmax_with_temperature(logits_array, 1.0)
    
    results = {
        'original_probabilities': original_probs,
        'calibrated_probabilities': None,
        'temperature': None,
        'validation_nll_before': None,
        'validation_nll_after': None
    }
    
    if groups is not None:
        # Group-wise temperature scaling
        unique_groups = np.unique(groups)
        group_temperatures = {}
        calibrated_probs = np.zeros_like(original_probs)
        
        for group in unique_groups:
            group_mask = groups == group
            group_logits = logits_array[group_mask]
            group_labels = true_labels[group_mask]
            
            if len(group_logits) < 10:  # Minimum samples for reliable optimization
                warnings.warn(f"Group '{group}' has only {len(group_logits)} samples. "
                             "Using temperature = 1.0")
                group_temp = 1.0
            else:
                # Split group data for validation
                train_idx, val_idx = train_test_split(
                    np.arange(len(group_logits)), 
                    test_size=validation_split,
                    random_state=random_state,
                    stratify=group_labels if len(np.unique(group_labels)) > 1 else None
                )
                
                train_logits = group_logits[train_idx]
                train_labels = group_labels[train_idx]
                val_logits = group_logits[val_idx]
                val_labels = group_labels[val_idx]
                
                group_temp = optimize_temperature(train_logits, train_labels, 
                                                val_logits, val_labels)
            
            group_temperatures[group] = group_temp
            calibrated_probs[group_mask] = softmax_with_temperature(group_logits, group_temp)
        
        results['group_temperatures'] = group_temperatures
        results['temperature'] = group_temperatures  # For backward compatibility
        results['calibrated_probabilities'] = calibrated_probs
        
    else:
        # Global temperature scaling
        # Split data for temperature optimization
        train_idx, val_idx = train_test_split(
            np.arange(n_samples),
            test_size=validation_split,
            random_state=random_state,
            stratify=true_labels if len(np.unique(true_labels)) > 1 else None
        )
        
        train_logits = logits_array[train_idx]
        train_labels = true_labels[train_idx]
        val_logits = logits_array[val_idx]
        val_labels = true_labels[val_idx]
        
        # Compute validation NLL before calibration
        val_probs_before = softmax_with_temperature(val_logits, 1.0)
        nll_before = negative_log_likelihood(val_probs_before, val_labels)
        
        # Optimize temperature
        optimal_temp = optimize_temperature(train_logits, train_labels, 
                                          val_logits, val_labels)
        
        # Compute validation NLL after calibration
        val_probs_after = softmax_with_temperature(val_logits, optimal_temp)
        nll_after = negative_log_likelihood(val_probs_after, val_labels)
        
        # Apply temperature scaling to all data
        calibrated_probs = softmax_with_temperature(logits_array, optimal_temp)
        
        results.update({
            'temperature': optimal_temp,
            'calibrated_probabilities': calibrated_probs,
            'validation_nll_before': nll_before,
            'validation_nll_after': nll_after
        })
    
    return results


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic logits (overconfident network)
    n_samples = 1000
    n_classes = 3
    
    # Create overconfident logits by scaling up random logits
    base_logits = np.random.randn(n_samples, n_classes)
    overconfident_logits = base_logits * 2.5  # Scale up to make overconfident
    
    # Generate true labels
    true_probs = np.exp(base_logits) / np.sum(np.exp(base_logits), axis=1, keepdims=True)
    true_labels = np.array([np.random.choice(n_classes, p=prob) for prob in true_probs])
    
    print("Temperature Scaling Example")
    print("=" * 40)
    
    # Apply temperature scaling
    results = temperature_scaling(
        logits=overconfident_logits,
        true_labels=true_labels,
        validation_split=0.2,
        random_state=42
    )
    
    print(f"Optimal temperature: {results['temperature']:.3f}")
    print(f"Validation NLL before calibration: {results['validation_nll_before']:.4f}")
    print(f"Validation NLL after calibration: {results['validation_nll_after']:.4f}")
    print(f"NLL improvement: {results['validation_nll_before'] - results['validation_nll_after']:.4f}")
    
    # Show probability changes for first few samples
    print("\nProbability changes (first 5 samples):")
    print("Original probabilities:")
    print(results['original_probabilities'][:5])
    print("Calibrated probabilities:")
    print(results['calibrated_probabilities'][:5])
    
    # Example with group-wise temperature scaling
    print("\n" + "=" * 40)
    print("Group-wise Temperature Scaling Example")
    print("=" * 40)
    
    # Create DataFrame with group information
    groups = np.random.choice(['A', 'B'], size=n_samples)
    logits_df = pd.DataFrame(overconfident_logits, columns=[f'logit_{i}' for i in range(n_classes)])
    logits_df['group'] = groups
    
    results_