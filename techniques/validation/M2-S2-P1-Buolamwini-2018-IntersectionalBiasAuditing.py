import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from itertools import product
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

def intersectional_bias_auditing(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    protected_attributes: Dict[str, Union[np.ndarray, List]],
    positive_label: Union[int, str] = 1,
    subgroup_threshold: int = 10
) -> Dict[str, Union[pd.DataFrame, Dict, float]]:
    """
    Perform intersectional bias auditing to identify performance disparities across
    intersectional subgroups defined by protected attributes.
    
    This technique, introduced by Buolamwini and Gebru (2018), evaluates model
    performance across intersectional demographic groups to uncover bias that
    may be hidden when examining single attributes in isolation.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels
    y_pred : array-like  
        Predicted binary labels from the model
    protected_attributes : dict
        Dictionary mapping attribute names to arrays of attribute values
        e.g., {'gender': ['M', 'F', 'M', ...], 'race': ['White', 'Black', ...]}
    positive_label : int or str, default=1
        Label considered as positive class
    subgroup_threshold : int, default=10
        Minimum number of samples required for a subgroup to be included in analysis
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'subgroup_metrics': DataFrame with performance metrics for each intersectional group
        - 'disparity_metrics': Dictionary with overall disparity measurements
        - 'largest_disparity': Dictionary identifying the largest performance gaps
        - 'subgroup_counts': DataFrame with sample counts per subgroup
        
    Raises
    ------
    ValueError
        If inputs have mismatched lengths or invalid values
    """
    
    # Input validation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if len(protected_attributes) == 0:
        raise ValueError("At least one protected attribute must be provided")
    
    # Validate protected attributes lengths
    n_samples = len(y_true)
    for attr_name, attr_values in protected_attributes.items():
        attr_values = np.array(attr_values)
        if len(attr_values) != n_samples:
            raise ValueError(f"Protected attribute '{attr_name}' length mismatch")
        protected_attributes[attr_name] = attr_values
    
    # Convert labels to binary if needed
    unique_labels = np.unique(y_true)
    if len(unique_labels) != 2:
        raise ValueError("Only binary classification is supported")
    
    # Ensure positive_label is in the data
    if positive_label not in unique_labels:
        raise ValueError(f"positive_label {positive_label} not found in y_true")
    
    # Create intersectional subgroups
    # Get all unique values for each attribute
    attribute_names = list(protected_attributes.keys())
    attribute_values = [np.unique(protected_attributes[attr]) for attr in attribute_names]
    
    # Generate all combinations of attribute values (Cartesian product)
    subgroup_combinations = list(product(*attribute_values))
    
    # Calculate metrics for each intersectional subgroup
    subgroup_results = []
    subgroup_counts = []
    
    for combination in subgroup_combinations:
        # Create mask for current subgroup
        mask = np.ones(n_samples, dtype=bool)
        subgroup_name_parts = []
        
        for i, (attr_name, attr_value) in enumerate(zip(attribute_names, combination)):
            mask &= (protected_attributes[attr_name] == attr_value)
            subgroup_name_parts.append(f"{attr_name}={attr_value}")
        
        subgroup_name = " & ".join(subgroup_name_parts)
        subgroup_size = np.sum(mask)
        
        # Store subgroup count
        count_row = {'subgroup': subgroup_name, 'count': subgroup_size}
        for attr_name, attr_value in zip(attribute_names, combination):
            count_row[attr_name] = attr_value
        subgroup_counts.append(count_row)
        
        # Skip subgroups with insufficient samples
        if subgroup_size < subgroup_threshold:
            continue
        
        # Extract subgroup predictions and labels
        y_true_sub = y_true[mask]
        y_pred_sub = y_pred[mask]
        
        # Calculate performance metrics
        metrics = _calculate_performance_metrics(y_true_sub, y_pred_sub, positive_label)
        
        # Add subgroup identification
        result_row = {'subgroup': subgroup_name, 'sample_size': subgroup_size}
        for attr_name, attr_value in zip(attribute_names, combination):
            result_row[attr_name] = attr_value
        result_row.update(metrics)
        
        subgroup_results.append(result_row)
    
    if len(subgroup_results) == 0:
        warnings.warn("No subgroups meet the minimum threshold requirement")
        return {
            'subgroup_metrics': pd.DataFrame(),
            'disparity_metrics': {},
            'largest_disparity': {},
            'subgroup_counts': pd.DataFrame(subgroup_counts)
        }
    
    # Convert results to DataFrame
    subgroup_df = pd.DataFrame(subgroup_results)
    counts_df = pd.DataFrame(subgroup_counts)
    
    # Calculate disparity metrics
    disparity_metrics = _calculate_disparity_metrics(subgroup_df)
    
    # Identify largest disparities
    largest_disparity = _identify_largest_disparities(subgroup_df)
    
    return {
        'subgroup_metrics': subgroup_df,
        'disparity_metrics': disparity_metrics,
        'largest_disparity': largest_disparity,
        'subgroup_counts': counts_df
    }

def _calculate_performance_metrics(y_true: np.ndarray, y_pred: np.ndarray, positive_label: Union[int, str]) -> Dict[str, float]:
    """
    Calculate standard binary classification performance metrics.
    
    Metrics calculated:
    - Accuracy: Overall correctness
    - TPR (Sensitivity/Recall): True Positive Rate
    - TNR (Specificity): True Negative Rate  
    - FPR: False Positive Rate
    - FNR: False Negative Rate
    - PPV (Precision): Positive Predictive Value
    - NPV: Negative Predictive Value
    """
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[positive_label, 1-positive_label if isinstance(positive_label, int) else None])
    
    # Handle case where confusion matrix might be 1x1 (only one class present)
    if cm.shape == (1, 1):
        # Only one class present in predictions
        if len(np.unique(y_true)) == 1:
            # Perfect prediction for single class
            accuracy = 1.0
            if y_true[0] == positive_label:
                return {
                    'accuracy': accuracy, 'tpr': 1.0, 'tnr': np.nan, 'fpr': np.nan, 
                    'fnr': 0.0, 'ppv': 1.0, 'npv': np.nan
                }
            else:
                return {
                    'accuracy': accuracy, 'tpr': np.nan, 'tnr': 1.0, 'fpr': 0.0,
                    'fnr': np.nan, 'ppv': np.nan, 'npv': 1.0
                }
    
    # Standard 2x2 confusion matrix
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback: manually calculate
        tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
        tn = np.sum((y_true != positive_label) & (y_pred != positive_label))
        fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
        fn = np.sum((y_true == positive_label) & (y_pred != positive_label))
    
    # Calculate metrics with division by zero handling
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # Sensitivity/Recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan  # Specificity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan  # False Negative Rate
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan  # Negative Predictive Value
    
    return {
        'accuracy': accuracy,
        'tpr': tpr,
        'tnr': tnr, 
        'fpr': fpr,
        'fnr': fnr,
        'ppv': ppv,
        'npv': npv
    }

def _calculate_disparity_metrics(subgroup_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate overall disparity metrics across all subgroups.
    
    Disparity is measured as the difference between maximum and minimum
    performance across subgroups for each metric.
    """
    
    metric_columns = ['accuracy', 'tpr', 'tnr', 'fpr', 'fnr', 'ppv', 'npv']
    disparity_metrics = {}
    
    for metric in metric_columns:
        if metric in subgroup_df.columns:
            values = subgroup_df[metric].dropna()
            if len(values) > 1:
                disparity_metrics[f'{metric}_disparity'] = values.max() - values.min()
                disparity_metrics[f'{metric}_ratio'] = values.min() / values.max() if values.max() > 0 else np.nan
                disparity_metrics[f'{metric}_std'] = values.std()
            else:
                disparity_metrics[f'{metric}_disparity'] = 0.0
                disparity_metrics[f'{metric}_ratio'] = 1.0
                disparity_metrics[f'{metric}_std'] = 0.0
    
    return disparity_metrics

def _identify_largest_disparities(subgroup_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Identify the subgroups with largest performance disparities.
    """
    
    metric_columns = ['accuracy', 'tpr', 'tnr', 'fpr', 'fnr', 'ppv', 'npv']
    largest_disparities = {}
    
    for metric in metric_columns:
        if metric in subgroup_df.columns:
            values = subgroup_df[metric].dropna()
            if len(values) > 1:
                max_idx = values.idxmax()
                min_idx = values.idxmin()
                
                largest