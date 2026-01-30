import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.optimize import minimize_scalar
import warnings

def group_specific_threshold_optimization(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    sensitive_groups: np.ndarray,
    constraint_type: str = 'equal_opportunity',
    objective: str = 'accuracy',
    grid_size: int = 100,
    tolerance: float = 0.01,
    random_state: Optional[int] = None
) -> Dict[str, Union[float, Dict, np.ndarray]]:
    """
    Optimize group-specific thresholds to satisfy fairness constraints.
    
    This implementation follows Hardt et al. (2016) approach for finding optimal
    thresholds that satisfy fairness constraints while maximizing a performance
    objective. The algorithm searches for intersection points of ROC curves
    across different demographic groups.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_scores : np.ndarray
        Predicted scores or probabilities from a classifier
    sensitive_groups : np.ndarray
        Group membership indicators (e.g., 0 for group A, 1 for group B)
    constraint_type : str, default='equal_opportunity'
        Type of fairness constraint:
        - 'equal_opportunity': Equal TPR across groups
        - 'equalized_odds': Equal TPR and FPR across groups
    objective : str, default='accuracy'
        Performance objective to maximize:
        - 'accuracy': Overall accuracy
        - 'balanced_accuracy': Balanced accuracy
    grid_size : int, default=100
        Number of grid points for constraint optimization
    tolerance : float, default=0.01
        Tolerance for constraint satisfaction
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Dict[str, Union[float, Dict, np.ndarray]]
        Dictionary containing:
        - 'optimal_thresholds': Dict mapping group to optimal threshold
        - 'constraint_value': Optimal constraint value (TPR for equal opportunity)
        - 'objective_value': Achieved objective value
        - 'group_metrics': Performance metrics by group
        - 'fairness_achieved': Whether constraints are satisfied
        - 'randomization_needed': Whether randomized predictors are needed
    """
    
    # Input validation
    if len(y_true) != len(y_scores) or len(y_true) != len(sensitive_groups):
        raise ValueError("All input arrays must have the same length")
    
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only binary values (0, 1)")
    
    if constraint_type not in ['equal_opportunity', 'equalized_odds']:
        raise ValueError("constraint_type must be 'equal_opportunity' or 'equalized_odds'")
    
    if objective not in ['accuracy', 'balanced_accuracy']:
        raise ValueError("objective must be 'accuracy' or 'balanced_accuracy'")
    
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    sensitive_groups = np.asarray(sensitive_groups)
    
    # Get unique groups
    unique_groups = np.unique(sensitive_groups)
    n_groups = len(unique_groups)
    
    if n_groups < 2:
        raise ValueError("At least two groups are required")
    
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Compute ROC curves for each group
    group_roc_data = {}
    for group in unique_groups:
        group_mask = sensitive_groups == group
        group_y_true = y_true[group_mask]
        group_y_scores = y_scores[group_mask]
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(group_y_true, group_y_scores)
        group_roc_data[group] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'y_true': group_y_true,
            'y_scores': group_y_scores
        }
    
    def _get_threshold_for_rate(group_data: Dict, target_rate: float, rate_type: str) -> float:
        """Find threshold that achieves target TPR or FPR for a group."""
        if rate_type == 'tpr':
            rates = group_data['tpr']
        else:  # fpr
            rates = group_data['fpr']
        
        # Find closest rate
        idx = np.argmin(np.abs(rates - target_rate))
        return group_data['thresholds'][idx]
    
    def _evaluate_predictions(y_true_group: np.ndarray, y_pred_group: np.ndarray) -> Dict:
        """Compute performance metrics for a group."""
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
        
        # Handle division by zero
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        balanced_acc = 0.5 * (tpr + (tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
        
        return {
            'tpr': tpr,
            'fpr': fpr,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }
    
    def _objective_function(constraint_value: float) -> Tuple[float, Dict]:
        """Evaluate objective for given constraint value."""
        group_thresholds = {}
        group_predictions = {}
        group_metrics = {}
        
        # Find thresholds for each group to satisfy constraint
        for group in unique_groups:
            group_data = group_roc_data[group]
            
            if constraint_type == 'equal_opportunity':
                # Find threshold that achieves target TPR
                threshold = _get_threshold_for_rate(group_data, constraint_value, 'tpr')
            else:  # equalized_odds - use TPR constraint (could extend to both TPR and FPR)
                threshold = _get_threshold_for_rate(group_data, constraint_value, 'tpr')
            
            group_thresholds[group] = threshold
            
            # Make predictions
            y_pred = (group_data['y_scores'] >= threshold).astype(int)
            group_predictions[group] = y_pred
            
            # Compute metrics
            group_metrics[group] = _evaluate_predictions(group_data['y_true'], y_pred)
        
        # Compute overall objective
        if objective == 'accuracy':
            total_correct = sum(np.sum(group_roc_data[g]['y_true'] == group_predictions[g]) 
                              for g in unique_groups)
            total_samples = len(y_true)
            obj_value = total_correct / total_samples
        else:  # balanced_accuracy
            group_balanced_accs = [group_metrics[g]['balanced_accuracy'] for g in unique_groups]
            obj_value = np.mean(group_balanced_accs)
        
        return obj_value, {
            'thresholds': group_thresholds,
            'predictions': group_predictions,
            'metrics': group_metrics
        }
    
    # Grid search over constraint values
    if constraint_type == 'equal_opportunity':
        # Search over TPR values
        constraint_grid = np.linspace(0.01, 0.99, grid_size)
    else:  # equalized_odds
        constraint_grid = np.linspace(0.01, 0.99, grid_size)
    
    best_objective = -np.inf
    best_constraint_value = None
    best_results = None
    
    for constraint_val in constraint_grid:
        try:
            obj_val, results = _objective_function(constraint_val)
            
            if obj_val > best_objective:
                best_objective = obj_val
                best_constraint_value = constraint_val
                best_results = results
                
        except (ValueError, IndexError):
            # Skip invalid constraint values
            continue
    
    if best_results is None:
        raise RuntimeError("No valid solution found")
    
    # Check if fairness constraints are satisfied
    group_tprs = [best_results['metrics'][g]['tpr'] for g in unique_groups]
    tpr_diff = np.max(group_tprs) - np.min(group_tprs)
    
    fairness_achieved = tpr_diff <= tolerance
    
    if constraint_type == 'equalized_odds':
        group_fprs = [best_results['metrics'][g]['fpr'] for g in unique_groups]
        fpr_diff = np.max(group_fprs) - np.min(group_fprs)
        fairness_achieved = fairness_achieved and (fpr_diff <= tolerance)
    
    # Check if randomization is needed (simplified check)
    randomization_needed = not fairness_achieved and tpr_diff > 2 * tolerance
    
    return {
        'optimal_thresholds': best_results['thresholds'],
        'constraint_value': best_constraint_value,
        'objective_value': best_objective,
        'group_metrics': best_results['metrics'],
        'fairness_achieved': fairness_achieved,
        'randomization_needed': randomization_needed,
        'tpr_difference': tpr_diff,
        'fpr_difference': np.max([best_results['metrics'][g]['fpr'] for g in unique_groups]) - 
                         np.min([best_results['metrics'][g]['fpr'] for g in unique_groups])
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic dataset
    n_samples = 1000
    
    # Create two groups with different base rates
    group_0_size = 600
    group_1_size = 400
    
    # Group 0 (majority group)
    y_true_0 = np.random.binomial(1, 0.3, group_0_size)  # 30% positive rate
    y_scores_0 = np.random.beta(2, 5, group_0_size)  # Lower scores on average
    y_scores_0[y_true_0 == 1] += np.random.normal(0.3, 0.1, np.sum(y_true_0 == 1))
    
    # Group 1 (minority group)  
    y_true_1 = np.random.binomial(1, 0.5, group_1_size)  # 50% positive rate
    y_scores_1 = np.random.beta(3, 4, group_1_size)  # Higher scores on average
    y_scores_1[y_true_1 == 1] += np.random.normal(0.2, 0.1, np.sum(y_true_1 == 1))
    
    # Combine data
    y_true = np.concatenate([y_true_0, y_true_1])
    y_scores = np.concatenate([y_scores_0, y_scores