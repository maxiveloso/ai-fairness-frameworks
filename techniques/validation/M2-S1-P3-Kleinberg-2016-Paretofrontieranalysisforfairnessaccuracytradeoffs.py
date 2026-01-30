import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Callable, Any
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings

def pareto_frontier_fairness_accuracy(
    models: List[BaseEstimator],
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    sensitive_attribute: Union[np.ndarray, pd.Series],
    accuracy_metric: str = 'accuracy',
    fairness_metrics: List[str] = ['demographic_parity', 'equalized_odds'],
    optimization_method: str = 'chebyshev',
    n_points: int = 50,
    return_dominated: bool = False
) -> Dict[str, Any]:
    """
    Analyze Pareto frontier for fairness-accuracy trade-offs in machine learning models.
    
    This function implements the theoretical framework from Kleinberg et al. (2016) for
    understanding inherent trade-offs between fairness and accuracy in algorithmic
    decision-making. The Pareto frontier represents the set of optimal solutions where
    improving one objective (fairness or accuracy) requires sacrificing the other.
    
    Parameters
    ----------
    models : List[BaseEstimator]
        List of trained scikit-learn compatible models to evaluate
    X_test : Union[np.ndarray, pd.DataFrame]
        Test features
    y_test : Union[np.ndarray, pd.Series]
        True test labels
    sensitive_attribute : Union[np.ndarray, pd.Series]
        Protected/sensitive attribute values (e.g., race, gender)
    accuracy_metric : str, default='accuracy'
        Accuracy metric to use ('accuracy', 'precision', 'recall')
    fairness_metrics : List[str], default=['demographic_parity', 'equalized_odds']
        List of fairness metrics to evaluate
    optimization_method : str, default='chebyshev'
        Method for Pareto optimization ('chebyshev', 'weighted_sum')
    n_points : int, default=50
        Number of points to generate for Pareto frontier approximation
    return_dominated : bool, default=False
        Whether to return dominated solutions in addition to Pareto optimal ones
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'pareto_frontier': DataFrame with Pareto optimal points
        - 'all_solutions': DataFrame with all evaluated solutions
        - 'dominated_solutions': DataFrame with dominated solutions (if return_dominated=True)
        - 'trade_off_analysis': Statistical analysis of trade-offs
        - 'optimal_models': Indices of models on Pareto frontier
        - 'fairness_accuracy_correlation': Correlation between fairness and accuracy
        - 'pareto_efficiency_ratio': Ratio of Pareto optimal to total solutions
    """
    
    # Input validation
    if not isinstance(models, list) or len(models) == 0:
        raise ValueError("models must be a non-empty list of trained models")
    
    if len(X_test) != len(y_test) or len(X_test) != len(sensitive_attribute):
        raise ValueError("X_test, y_test, and sensitive_attribute must have same length")
    
    if accuracy_metric not in ['accuracy', 'precision', 'recall']:
        raise ValueError("accuracy_metric must be one of: 'accuracy', 'precision', 'recall'")
    
    valid_fairness_metrics = ['demographic_parity', 'equalized_odds', 'equal_opportunity']
    for metric in fairness_metrics:
        if metric not in valid_fairness_metrics:
            raise ValueError(f"fairness_metric {metric} not supported. Use: {valid_fairness_metrics}")
    
    # Convert inputs to numpy arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    sensitive_attribute = np.array(sensitive_attribute)
    
    # Get unique groups in sensitive attribute
    groups = np.unique(sensitive_attribute)
    if len(groups) != 2:
        warnings.warn("Analysis optimized for binary sensitive attributes")
    
    def _calculate_accuracy_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specified accuracy metric"""
        if accuracy_metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif accuracy_metric == 'precision':
            return precision_score(y_true, y_pred, average='weighted', zero_division=0)
        elif accuracy_metric == 'recall':
            return recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    def _calculate_demographic_parity(y_pred: np.ndarray, sensitive_attr: np.ndarray) -> float:
        """
        Calculate demographic parity violation.
        Measures difference in positive prediction rates between groups.
        Lower values indicate better fairness.
        """
        group_rates = []
        for group in groups:
            group_mask = sensitive_attr == group
            if np.sum(group_mask) > 0:
                positive_rate = np.mean(y_pred[group_mask])
                group_rates.append(positive_rate)
        
        if len(group_rates) < 2:
            return 0.0
        
        # Return absolute difference (violation measure)
        return abs(group_rates[0] - group_rates[1])
    
    def _calculate_equalized_odds(y_true: np.ndarray, y_pred: np.ndarray, 
                                sensitive_attr: np.ndarray) -> float:
        """
        Calculate equalized odds violation.
        Measures difference in true positive rates between groups.
        Lower values indicate better fairness.
        """
        group_tpr = []
        for group in groups:
            group_mask = sensitive_attr == group
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]
            
            if np.sum(y_true_group) > 0:  # Avoid division by zero
                tpr = np.sum((y_true_group == 1) & (y_pred_group == 1)) / np.sum(y_true_group == 1)
                group_tpr.append(tpr)
        
        if len(group_tpr) < 2:
            return 0.0
        
        return abs(group_tpr[0] - group_tpr[1])
    
    def _calculate_equal_opportunity(y_true: np.ndarray, y_pred: np.ndarray,
                                   sensitive_attr: np.ndarray) -> float:
        """
        Calculate equal opportunity violation.
        Same as equalized odds for positive class only.
        """
        return _calculate_equalized_odds(y_true, y_pred, sensitive_attr)
    
    def _calculate_fairness_metric(metric_name: str, y_true: np.ndarray, 
                                 y_pred: np.ndarray, sensitive_attr: np.ndarray) -> float:
        """Calculate specified fairness metric"""
        if metric_name == 'demographic_parity':
            return _calculate_demographic_parity(y_pred, sensitive_attr)
        elif metric_name == 'equalized_odds':
            return _calculate_equalized_odds(y_true, y_pred, sensitive_attr)
        elif metric_name == 'equal_opportunity':
            return _calculate_equal_opportunity(y_true, y_pred, sensitive_attr)
        else:
            raise ValueError(f"Unknown fairness metric: {metric_name}")
    
    # Evaluate all models
    results = []
    for i, model in enumerate(models):
        try:
            # Get predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
            else:
                raise ValueError(f"Model {i} does not have predict method")
            
            # Calculate accuracy
            accuracy = _calculate_accuracy_metric(y_test, y_pred)
            
            # Calculate fairness metrics
            fairness_scores = {}
            for fairness_metric in fairness_metrics:
                fairness_violation = _calculate_fairness_metric(
                    fairness_metric, y_test, y_pred, sensitive_attribute
                )
                fairness_scores[fairness_metric] = fairness_violation
            
            result = {
                'model_index': i,
                'accuracy': accuracy,
                **fairness_scores
            }
            results.append(result)
            
        except Exception as e:
            warnings.warn(f"Error evaluating model {i}: {str(e)}")
            continue
    
    if not results:
        raise ValueError("No models could be successfully evaluated")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    def _is_pareto_optimal(costs: np.ndarray, maximize_objectives: List[bool]) -> np.ndarray:
        """
        Identify Pareto optimal solutions.
        For fairness-accuracy trade-off: maximize accuracy, minimize fairness violations.
        """
        n_points = costs.shape[0]
        is_efficient = np.ones(n_points, dtype=bool)
        
        for i in range(n_points):
            if is_efficient[i]:
                # Compare with all other points
                for j in range(n_points):
                    if i != j and is_efficient[j]:
                        dominates = True
                        strictly_better = False
                        
                        for k, maximize in enumerate(maximize_objectives):
                            if maximize:
                                # For accuracy (maximize): j dominates i if j >= i
                                if costs[j, k] < costs[i, k]:
                                    dominates = False
                                    break
                                elif costs[j, k] > costs[i, k]:
                                    strictly_better = True
                            else:
                                # For fairness violations (minimize): j dominates i if j <= i
                                if costs[j, k] > costs[i, k]:
                                    dominates = False
                                    break
                                elif costs[j, k] < costs[i, k]:
                                    strictly_better = True
                        
                        # j dominates i if j is better or equal in all objectives
                        # and strictly better in at least one
                        if dominates and strictly_better:
                            is_efficient[i] = False
                            break
        
        return is_efficient
    
    # Prepare cost matrix for Pareto analysis
    # Columns: accuracy (maximize), fairness violations (minimize)
    cost_matrix = results_df[['accuracy'] + fairness_metrics].values
    
    # Define which objectives to maximize (True) or minimize (False)
    maximize_objectives = [True] + [False] * len(fairness_metrics)
    
    # Find Pareto optimal solutions
    pareto_mask = _is_pareto_optimal(cost_matrix, maximize_objectives)
    pareto_frontier = results_df[pareto_mask].copy()
    dominated_solutions = results_df[~pareto_mask].copy()
    
    # Trade-off analysis
    trade_off_stats = {}
    
    # Calculate correlations between accuracy and fairness metrics
    correlations = {}
    for fairness_metric in fairness_metrics:
        corr = results_df['accuracy'].corr(results_df[fairness_metric])
        correlations[f'accuracy_{fairness_metric}_correlation'] = corr
    
    # Calculate Pareto efficiency ratio
    pareto_efficiency_ratio = len(pareto_frontier) / len(results_df)
    
    # Statistical analysis of trade-offs
    for fairness_metric in fairness_metrics:
        # Range analysis
        accuracy_range = results_df['accuracy'].max() - results_df['accuracy'].min()
        fairness_range