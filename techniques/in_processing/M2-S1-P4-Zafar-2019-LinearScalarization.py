import numpy as np
import pandas as pd
from typing import List, Callable, Dict, Any, Optional, Union
from scipy.optimize import minimize
import warnings

def linear_scalarization(
    objective_functions: List[Callable[[np.ndarray], float]],
    weights: Union[np.ndarray, List[float]],
    initial_params: np.ndarray,
    bounds: Optional[List[tuple]] = None,
    method: str = 'L-BFGS-B',
    explore_pareto: bool = False,
    n_pareto_points: int = 10,
    **optimizer_kwargs
) -> Dict[str, Any]:
    """
    Implement Linear Scalarization for multi-objective optimization.
    
    Linear scalarization converts a multi-objective optimization problem into a 
    single-objective problem by taking a weighted linear combination of all objectives.
    The scalarized objective is: L(θ) = Σ λi * fi(θ), where λi are weights and 
    fi(θ) are individual objective functions.
    
    This technique is particularly useful in fair machine learning where one might
    want to balance performance and fairness objectives.
    
    Parameters
    ----------
    objective_functions : List[Callable[[np.ndarray], float]]
        List of objective functions to be minimized. Each function should take
        a parameter vector and return a scalar value.
    weights : Union[np.ndarray, List[float]]
        Weight vector λ with non-negative values that sum to 1. Length must
        match the number of objective functions.
    initial_params : np.ndarray
        Initial parameter values for optimization.
    bounds : Optional[List[tuple]], default=None
        Bounds for each parameter as list of (min, max) tuples.
    method : str, default='L-BFGS-B'
        Optimization method to use (from scipy.optimize.minimize).
    explore_pareto : bool, default=False
        Whether to systematically vary weights to explore Pareto frontier.
    n_pareto_points : int, default=10
        Number of points to sample along Pareto frontier if explore_pareto=True.
    **optimizer_kwargs
        Additional keyword arguments passed to scipy.optimize.minimize.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'optimal_params': Optimized parameters for given weights
        - 'optimal_value': Value of scalarized objective at optimum
        - 'individual_objectives': Values of each objective at optimum
        - 'weights_used': Weight vector used
        - 'optimization_success': Whether optimization converged
        - 'optimization_message': Optimization status message
        - 'pareto_frontier': If explore_pareto=True, array of Pareto optimal solutions
        - 'pareto_objectives': If explore_pareto=True, objective values for Pareto points
    
    Examples
    --------
    >>> import numpy as np
    >>> 
    >>> # Define two competing objectives
    >>> def f1(x):  # Minimize sum of squares
    ...     return np.sum(x**2)
    >>> 
    >>> def f2(x):  # Minimize negative sum (maximize sum)
    ...     return -np.sum(x)
    >>> 
    >>> objectives = [f1, f2]
    >>> weights = [0.7, 0.3]  # Prioritize f1 over f2
    >>> initial = np.array([1.0, 1.0])
    >>> 
    >>> result = linear_scalarization(objectives, weights, initial)
    >>> print(f"Optimal parameters: {result['optimal_params']}")
    >>> print(f"Individual objectives: {result['individual_objectives']}")
    """
    
    # Input validation
    if not isinstance(objective_functions, list) or len(objective_functions) == 0:
        raise ValueError("objective_functions must be a non-empty list of callable functions")
    
    if not all(callable(f) for f in objective_functions):
        raise ValueError("All elements in objective_functions must be callable")
    
    weights = np.asarray(weights, dtype=float)
    
    if len(weights) != len(objective_functions):
        raise ValueError("Length of weights must match number of objective functions")
    
    if np.any(weights < 0):
        raise ValueError("All weights must be non-negative")
    
    if not np.isclose(np.sum(weights), 1.0, rtol=1e-10):
        warnings.warn("Weights do not sum to 1, normalizing automatically")
        weights = weights / np.sum(weights)
    
    initial_params = np.asarray(initial_params, dtype=float)
    
    if initial_params.ndim != 1:
        raise ValueError("initial_params must be a 1-dimensional array")
    
    # Test objective functions with initial parameters
    try:
        test_values = [f(initial_params) for f in objective_functions]
        if not all(np.isfinite(val) and np.isscalar(val) for val in test_values):
            raise ValueError("Objective functions must return finite scalar values")
    except Exception as e:
        raise ValueError(f"Error evaluating objective functions: {str(e)}")
    
    def scalarized_objective(params):
        """
        Compute the scalarized objective L(θ) = Σ λi * fi(θ)
        
        This combines multiple objectives into a single objective using
        the provided weight vector.
        """
        try:
            individual_values = np.array([f(params) for f in objective_functions])
            return np.dot(weights, individual_values)
        except Exception:
            return np.inf  # Return large value if evaluation fails
    
    # Perform single optimization with given weights
    try:
        result = minimize(
            scalarized_objective,
            initial_params,
            method=method,
            bounds=bounds,
            **optimizer_kwargs
        )
        
        optimal_params = result.x
        optimal_value = result.fun
        success = result.success
        message = result.message
        
    except Exception as e:
        optimal_params = initial_params.copy()
        optimal_value = scalarized_objective(initial_params)
        success = False
        message = f"Optimization failed: {str(e)}"
    
    # Evaluate individual objectives at optimum
    individual_objectives = np.array([f(optimal_params) for f in objective_functions])
    
    # Prepare results dictionary
    results = {
        'optimal_params': optimal_params,
        'optimal_value': optimal_value,
        'individual_objectives': individual_objectives,
        'weights_used': weights,
        'optimization_success': success,
        'optimization_message': message
    }
    
    # Explore Pareto frontier if requested
    if explore_pareto:
        pareto_solutions = []
        pareto_objectives = []
        
        # Generate systematic weight variations
        if len(objective_functions) == 2:
            # For 2 objectives, vary first weight from 0 to 1
            weight_values = np.linspace(0, 1, n_pareto_points)
            weight_combinations = [[w, 1-w] for w in weight_values]
        else:
            # For >2 objectives, use random sampling from simplex
            # Generate random points on simplex using Dirichlet distribution
            weight_combinations = np.random.dirichlet(
                np.ones(len(objective_functions)), 
                size=n_pareto_points
            )
        
        for weight_combo in weight_combinations:
            weight_combo = np.asarray(weight_combo)
            
            def temp_objective(params):
                individual_values = np.array([f(params) for f in objective_functions])
                return np.dot(weight_combo, individual_values)
            
            try:
                temp_result = minimize(
                    temp_objective,
                    initial_params,
                    method=method,
                    bounds=bounds,
                    **optimizer_kwargs
                )
                
                if temp_result.success:
                    pareto_solutions.append(temp_result.x)
                    pareto_obj_values = [f(temp_result.x) for f in objective_functions]
                    pareto_objectives.append(pareto_obj_values)
                    
            except Exception:
                continue  # Skip failed optimizations
        
        if pareto_solutions:
            results['pareto_frontier'] = np.array(pareto_solutions)
            results['pareto_objectives'] = np.array(pareto_objectives)
        else:
            results['pareto_frontier'] = np.array([])
            results['pareto_objectives'] = np.array([])
    
    return results


if __name__ == "__main__":
    # Example 1: Simple bi-objective optimization
    print("Example 1: Bi-objective optimization")
    print("=" * 50)
    
    def objective1(x):
        """Minimize sum of squares (prefer small values)"""
        return np.sum(x**2)
    
    def objective2(x):
        """Minimize negative sum (maximize sum, prefer large values)"""
        return -np.sum(x)
    
    objectives = [objective1, objective2]
    weights = [0.8, 0.2]  # Prioritize objective1
    initial = np.array([2.0, 3.0])
    
    result1 = linear_scalarization(
        objectives, 
        weights, 
        initial,
        bounds=[(-5, 5), (-5, 5)]
    )
    
    print(f"Weights: {result1['weights_used']}")
    print(f"Optimal parameters: {result1['optimal_params']}")
    print(f"Scalarized objective value: {result1['optimal_value']:.4f}")
    print(f"Individual objectives: {result1['individual_objectives']}")
    print(f"Optimization successful: {result1['optimization_success']}")
    print()
    
    # Example 2: Fairness-Performance tradeoff simulation
    print("Example 2: Fairness-Performance tradeoff")
    print("=" * 50)
    
    def performance_loss(theta):
        """Simulate classification performance loss (to minimize)"""
        # Simulate that performance degrades as we move away from [1, 1]
        return np.sum((theta - np.array([1.0, 1.0]))**2)
    
    def fairness_violation(theta):
        """Simulate fairness constraint violation (to minimize)"""
        # Simulate that fairness improves as we approach [0, 0]
        return np.sum(theta**2)
    
    fairness_objectives = [performance_loss, fairness_violation]
    
    # Test different fairness-performance tradeoffs
    tradeoff_scenarios = [
        ([0.9, 0.1], "Performance-focused"),
        ([0.5, 0.5], "Balanced"),
        ([0.1, 0.9], "Fairness-focused")
    ]
    
    for weights, scenario_name in tradeoff_scenarios:
        result = linear_scalarization(
            fairness_objectives,
            weights,
            np.array([0.5, 0.5]),
            bounds=[(0, 2), (0, 2)]
        )
        
        print(f"{scenario_name} (weights: {weights}):")
        print(f"  Optimal θ: {result['optimal_params']}")
        print(f"  Performance loss: {result['individual_objectives'][0]:.4f}")
        print(f"  Fairness violation: {result['individual_objectives'][1]:.4f}")
        print()
    
    # Example 3: Pareto frontier exploration
    print("Example 3: Pareto frontier exploration")
    print("=" * 50)
    
    pareto_result = linear_scalarization(
        fairness_objectives,
        [0.5, 0.5],  # Starting weights
        np.array([0.