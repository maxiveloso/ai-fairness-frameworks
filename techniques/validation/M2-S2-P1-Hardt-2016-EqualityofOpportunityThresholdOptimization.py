import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.optimize import minimize_scalar
from sklearn.metrics import confusion_matrix
import warnings

def equality_of_opportunity_threshold_optimization(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    sensitive_attribute: np.ndarray,
    method: str = 'fixed',
    base_rate_constraint: bool = True,
    tolerance: float = 1e-6,
    max_iterations: int = 100
) -> Dict[str, Union[float, Dict, np.ndarray]]:
    """
    Optimize classification thresholds to achieve equality of opportunity.
    
    Equality of opportunity requires that the true positive rate (TPR) is equal
    across different groups defined by a sensitive attribute. This post-processing
    method adjusts decision thresholds to satisfy this fairness constraint while
    minimizing classification error.
    
    The optimization can use either:
    1. Fixed thresholds: Each group gets a single threshold
    2. Randomized thresholds: Each group uses a mixture of two thresholds
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_scores : np.ndarray
        Predicted scores/probabilities from classifier
    sensitive_attribute : np.ndarray
        Binary sensitive attribute defining groups (0 or 1)
    method : str, default='fixed'
        Threshold optimization method ('fixed' or 'randomized')
    base_rate_constraint : bool, default=True
        Whether to maintain overall base rate (positive prediction rate)
    tolerance : float, default=1e-6
        Convergence tolerance for optimization
    max_iterations : int, default=100
        Maximum iterations for optimization
        
    Returns
    -------
    Dict[str, Union[float, Dict, np.ndarray]]
        Dictionary containing:
        - 'optimal_thresholds': Optimized thresholds for each group
        - 'tpr_before': True positive rates before optimization
        - 'tpr_after': True positive rates after optimization
        - 'fpr_before': False positive rates before optimization  
        - 'fpr_after': False positive rates after optimization
        - 'accuracy_before': Accuracy before optimization
        - 'accuracy_after': Accuracy after optimization
        - 'equalized_odds_violation_before': EO violation before
        - 'equalized_odds_violation_after': EO violation after
        - 'base_rate_before': Base rates before optimization
        - 'base_rate_after': Base rates after optimization
        - 'randomization_weights': Mixing weights (if randomized method)
        
    Raises
    ------
    ValueError
        If inputs have incompatible shapes or invalid values
    """
    
    # Input validation
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    sensitive_attribute = np.asarray(sensitive_attribute)
    
    if len(y_true) != len(y_scores) or len(y_true) != len(sensitive_attribute):
        raise ValueError("All input arrays must have the same length")
    
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1")
        
    if not np.all(np.isin(sensitive_attribute, [0, 1])):
        raise ValueError("sensitive_attribute must contain only 0 and 1")
        
    if method not in ['fixed', 'randomized']:
        raise ValueError("method must be 'fixed' or 'randomized'")
    
    n_samples = len(y_true)
    groups = np.unique(sensitive_attribute)
    
    # Compute initial statistics with default threshold of 0.5
    initial_threshold = 0.5
    y_pred_initial = (y_scores >= initial_threshold).astype(int)
    
    def compute_group_statistics(y_true, y_pred, sensitive_attr):
        """Compute TPR, FPR, and other metrics for each group"""
        stats = {}
        for group in [0, 1]:
            mask = sensitive_attr == group
            if np.sum(mask) == 0:
                continue
                
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # Confusion matrix elements
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1]).ravel()
            
            # Rates with division by zero protection
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            accuracy = (tp + tn) / len(y_true_group) if len(y_true_group) > 0 else 0.0
            base_rate = np.mean(y_pred_group)
            
            stats[group] = {
                'tpr': tpr,
                'fpr': fpr, 
                'accuracy': accuracy,
                'base_rate': base_rate,
                'n_samples': len(y_true_group),
                'n_positive': tp + fn,
                'n_negative': tn + fp
            }
        
        return stats
    
    # Initial statistics
    initial_stats = compute_group_statistics(y_true, y_pred_initial, sensitive_attribute)
    
    def objective_function(thresholds, randomization_weights=None):
        """
        Objective function for threshold optimization.
        Minimizes classification error while satisfying equality of opportunity.
        """
        if method == 'fixed':
            threshold_0, threshold_1 = thresholds
            y_pred_opt = np.zeros_like(y_true)
            
            # Apply thresholds to each group
            mask_0 = sensitive_attribute == 0
            mask_1 = sensitive_attribute == 1
            
            y_pred_opt[mask_0] = (y_scores[mask_0] >= threshold_0).astype(int)
            y_pred_opt[mask_1] = (y_scores[mask_1] >= threshold_1).astype(int)
            
        else:  # randomized
            # For randomized method, use mixture of two thresholds per group
            threshold_0_low, threshold_0_high, threshold_1_low, threshold_1_high = thresholds
            weight_0, weight_1 = randomization_weights
            
            y_pred_opt = np.zeros_like(y_true)
            
            # Group 0 randomization
            mask_0 = sensitive_attribute == 0
            rand_0 = np.random.random(np.sum(mask_0))
            pred_0_low = (y_scores[mask_0] >= threshold_0_low).astype(int)
            pred_0_high = (y_scores[mask_0] >= threshold_0_high).astype(int)
            y_pred_opt[mask_0] = np.where(rand_0 < weight_0, pred_0_low, pred_0_high)
            
            # Group 1 randomization  
            mask_1 = sensitive_attribute == 1
            rand_1 = np.random.random(np.sum(mask_1))
            pred_1_low = (y_scores[mask_1] >= threshold_1_low).astype(int)
            pred_1_high = (y_scores[mask_1] >= threshold_1_high).astype(int)
            y_pred_opt[mask_1] = np.where(rand_1 < weight_1, pred_1_low, pred_1_high)
        
        # Compute statistics for optimization
        stats = compute_group_statistics(y_true, y_pred_opt, sensitive_attribute)
        
        # Equality of opportunity violation (TPR difference)
        if 0 in stats and 1 in stats:
            eo_violation = abs(stats[0]['tpr'] - stats[1]['tpr'])
        else:
            eo_violation = 0.0
        
        # Classification error
        accuracy = np.mean(y_true == y_pred_opt)
        error = 1 - accuracy
        
        # Base rate constraint violation
        base_rate_violation = 0.0
        if base_rate_constraint:
            original_base_rate = np.mean(y_pred_initial)
            current_base_rate = np.mean(y_pred_opt)
            base_rate_violation = abs(original_base_rate - current_base_rate)
        
        # Combined objective: minimize error + penalty for fairness violation
        penalty_weight = 1000.0  # Large penalty for fairness violations
        objective = error + penalty_weight * eo_violation + penalty_weight * base_rate_violation
        
        return objective, eo_violation, error, stats
    
    # Optimization using ternary search approach
    if method == 'fixed':
        # Grid search over threshold pairs
        threshold_range = np.linspace(0.01, 0.99, 50)
        best_objective = float('inf')
        best_thresholds = (0.5, 0.5)
        best_stats = None
        
        for t0 in threshold_range:
            for t1 in threshold_range:
                obj_val, eo_viol, error, stats = objective_function((t0, t1))
                
                if obj_val < best_objective:
                    best_objective = obj_val
                    best_thresholds = (t0, t1)
                    best_stats = stats
                    
    else:  # randomized method
        # Simplified randomized approach - use grid search with random weights
        threshold_range = np.linspace(0.01, 0.99, 20)
        weight_range = np.linspace(0.1, 0.9, 10)
        
        best_objective = float('inf')
        best_thresholds = (0.3, 0.7, 0.3, 0.7)
        best_weights = (0.5, 0.5)
        best_stats = None
        
        for t0_low in threshold_range[::2]:
            for t0_high in threshold_range[::2]:
                for t1_low in threshold_range[::2]:
                    for t1_high in threshold_range[::2]:
                        for w0 in weight_range[::2]:
                            for w1 in weight_range[::2]:
                                if t0_low <= t0_high and t1_low <= t1_high:
                                    thresholds = (t0_low, t0_high, t1_low, t1_high)
                                    weights = (w0, w1)
                                    
                                    obj_val, eo_viol, error, stats = objective_function(
                                        thresholds, weights
                                    )
                                    
                                    if obj_val < best_objective:
                                        best_objective = obj_val
                                        best_thresholds = thresholds
                                        best_weights = weights
                                        best_stats = stats
    
    # Generate final predictions with optimal thresholds
    if method == 'fixed':
        y_pred_final = np.zeros_like(y_true)
        mask_0 = sensitive_attribute == 0
        mask_1 = sensitive_attribute == 1
        
        y_pred_final[mask_0] = (y_scores[mask_0] >= best_thresholds[0]).astype(int)
        y_pred_final[mask_1] = (y_scores[mask_1] >= best_thresholds[1]).astype(int)
        
    else:  # randomized
        y_pred_final = np.zeros_like(y_true)
        mask_0 = sensitive_attribute == 0
        mask_1 = sensitive_attribute == 1

        # Apply randomized thresholds with weights
        t0_low, t0_high, t1_low, t1_high = best_thresholds
        w0, w1 = best_weights

        # For group 0: randomize between low and high threshold
        for i in np.where(mask_0)[0]:
            if np.random.random() < w0:
                y_pred_final[i] = 1 if y_scores[i] >= t0_low else 0
            else:
                y_pred_final[i] = 1 if y_scores[i] >= t0_high else 0

        # For group 1: randomize between low and high threshold
        for i in np.where(mask_1)[0]:
            if np.random.random() < w1:
                y_pred_final[i] = 1 if y_scores[i] >= t1_low else 0
            else:
                y_pred_final[i] = 1 if y_scores[i] >= t1_high else 0

    # Calculate final statistics
    initial_pred = (y_scores >= 0.5).astype(int)
    initial_stats = calculate_fairness_stats(y_true, initial_pred, sensitive_attribute)
    final_stats = calculate_fairness_stats(y_true, y_pred_final, sensitive_attribute)

    result = {
        'optimal_thresholds': best_thresholds,
        'tpr_before': initial_stats['tpr'],
        'tpr_after': final_stats['tpr'],
        'fpr_before': initial_stats['fpr'],
        'fpr_after': final_stats['fpr'],
        'accuracy_before': initial_stats['accuracy'],
        'accuracy_after': final_stats['accuracy'],
        'equalized_odds_violation_before': initial_stats['eo_violation'],
        'equalized_odds_violation_after': final_stats['eo_violation'],
        'base_rate_before': initial_stats['base_rate'],
        'base_rate_after': final_stats['base_rate'],
        'y_pred_adjusted': y_pred_final
    }

    if method == 'randomized':
        result['randomization_weights'] = best_weights

    return result

# ============================================================================
# CLI WRAPPER (added for case study execution)
# ============================================================================

def main():
    import argparse
    import json
    import sys
    
    parser = argparse.ArgumentParser(
        description="Equality of Opportunity Threshold Optimization - Hardt et al. (2016)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--data", required=True, help="Path to CSV with predictions and labels")
    parser.add_argument("--protected", required=True, help="Protected attribute column name")
    parser.add_argument("--outcome", required=True, help="True outcome column name")
    parser.add_argument("--scores", required=True, help="Prediction scores column name")
    parser.add_argument("--constraint", default="equalized_odds", 
                        choices=["equalized_odds", "equal_opportunity", "demographic_parity"],
                        help="Fairness constraint to optimize (default: equalized_odds)")
    parser.add_argument("--output", default="optimized_predictions.csv", help="Output file path")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    
    args = parser.parse_args()
    
    data = pd.read_csv(args.data)
    
    # Run technique (adjust function name as needed)
    result = equality_opportunity_threshold_optimization(
        scores=data[args.scores].values,
        y_true=data[args.outcome].values,
        protected=data[args.protected].values,
        constraint=args.constraint
    )
    
    # Save optimized predictions
    if 'optimized_predictions' in result:
        data['optimized_pred'] = result['optimized_predictions']
        data.to_csv(args.output, index=False)
    
    if args.json:
        output = {
            "technique": "Equality of Opportunity Threshold Optimization",
            "technique_id": 13,
            "citation": "Hardt et al. (2016)",
            "parameters": {
                "protected_attribute": args.protected,
                "constraint": args.constraint,
            },
            "results": {
                "thresholds_per_group": result.get('thresholds'),
                "equalized_odds_before": result.get('eo_before'),
                "equalized_odds_after": result.get('eo_after'),
                "accuracy_before": result.get('accuracy_before'),
                "accuracy_after": result.get('accuracy_after'),
            },
            "output_file": args.output
        }
        print(json.dumps(output, indent=2, default=str))
    elif args.verbose:
        print(f"\nThreshold Optimization Results:")
        print(f"  Constraint: {args.constraint}")
        print(f"  Thresholds: {result.get('thresholds')}")
        print(f"  EO before: {result.get('eo_before')}")
        print(f"  EO after: {result.get('eo_after')}")


if __name__ == "__main__":
    main()
