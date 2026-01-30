import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Union, Tuple, Any
from scipy.optimize import minimize
import warnings

def proxy_lagrangian_formulation(
    objective_func: Callable[[np.ndarray], float],
    objective_grad: Callable[[np.ndarray], np.ndarray],
    proxy_constraints: List[Callable[[np.ndarray], float]],
    proxy_constraint_grads: List[Callable[[np.ndarray], np.ndarray]],
    original_constraints: List[Callable[[np.ndarray], float]],
    x_init: np.ndarray,
    constraint_bounds: List[float],
    max_iterations: int = 1000,
    learning_rate_primal: float = 0.01,
    learning_rate_dual: float = 0.1,
    penalty_update_rate: float = 1.1,
    tolerance: float = 1e-6,
    max_penalty: float = 1000.0,
    regret_type: str = "external",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Implements the Proxy-Lagrangian Formulation for optimization with non-differentiable constraints.
    
    This method formulates constrained optimization as a two-player game where:
    - Player 1 (primal): Minimizes external regret using differentiable proxy constraints
    - Player 2 (dual): Enforces original constraints by minimizing swap regret
    
    The algorithm uses proxy constraints as differentiable approximations of non-differentiable
    original constraints, with penalty magnitudes automatically adjusted during optimization.
    
    Parameters
    ----------
    objective_func : Callable[[np.ndarray], float]
        The objective function to minimize
    objective_grad : Callable[[np.ndarray], np.ndarray]
        Gradient of the objective function
    proxy_constraints : List[Callable[[np.ndarray], float]]
        List of differentiable proxy constraint functions (should be <= 0 when satisfied)
    proxy_constraint_grads : List[Callable[[np.ndarray], np.ndarray]]
        List of gradients for proxy constraints
    original_constraints : List[Callable[[np.ndarray], float]]
        List of original (possibly non-differentiable) constraints
    x_init : np.ndarray
        Initial point for optimization
    constraint_bounds : List[float]
        Upper bounds for constraint violations
    max_iterations : int, default=1000
        Maximum number of iterations
    learning_rate_primal : float, default=0.01
        Learning rate for primal variables
    learning_rate_dual : float, default=0.1
        Learning rate for dual variables (Lagrange multipliers)
    penalty_update_rate : float, default=1.1
        Rate at which penalty parameters are updated
    tolerance : float, default=1e-6
        Convergence tolerance
    max_penalty : float, default=1000.0
        Maximum penalty parameter value
    regret_type : str, default="external"
        Type of regret minimization ("external" or "swap")
    verbose : bool, default=False
        Whether to print optimization progress
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'optimal_x': Optimal solution
        - 'optimal_objective': Optimal objective value
        - 'constraint_violations': Final constraint violations
        - 'lagrange_multipliers': Final dual variables
        - 'penalty_parameters': Final penalty parameters
        - 'convergence_history': History of objective and constraint values
        - 'converged': Whether algorithm converged
        - 'iterations': Number of iterations performed
        - 'final_regret': Final regret values
        
    Raises
    ------
    ValueError
        If input dimensions are inconsistent or parameters are invalid
    """
    
    # Input validation
    if not callable(objective_func) or not callable(objective_grad):
        raise ValueError("Objective function and gradient must be callable")
    
    if len(proxy_constraints) != len(proxy_constraint_grads):
        raise ValueError("Number of proxy constraints must match number of gradients")
    
    if len(proxy_constraints) != len(original_constraints):
        raise ValueError("Number of proxy and original constraints must match")
    
    if len(constraint_bounds) != len(proxy_constraints):
        raise ValueError("Number of constraint bounds must match number of constraints")
    
    if not isinstance(x_init, np.ndarray):
        x_init = np.array(x_init)
    
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    
    if learning_rate_primal <= 0 or learning_rate_dual <= 0:
        raise ValueError("Learning rates must be positive")
    
    if penalty_update_rate <= 1.0:
        raise ValueError("Penalty update rate must be greater than 1.0")
    
    # Initialize variables
    n_vars = len(x_init)
    n_constraints = len(proxy_constraints)
    
    # Primal variables
    x = x_init.copy()
    
    # Dual variables (Lagrange multipliers)
    lambdas = np.zeros(n_constraints)
    
    # Penalty parameters for proxy-original constraint discrepancy
    penalty_params = np.ones(n_constraints)
    
    # History tracking
    history = {
        'objective': [],
        'constraint_violations': [],
        'proxy_violations': [],
        'regret': [],
        'penalty_params': []
    }
    
    # Regret tracking for both players
    primal_regret = 0.0
    dual_regret = 0.0
    
    converged = False
    
    for iteration in range(max_iterations):
        # Evaluate current state
        obj_val = objective_func(x)
        obj_grad = objective_grad(x)
        
        # Evaluate constraints
        proxy_viols = np.array([max(0, c(x)) for c in proxy_constraints])
        original_viols = np.array([max(0, c(x)) for c in original_constraints])
        
        # Evaluate proxy constraint gradients
        proxy_grads = np.array([grad(x) for grad in proxy_constraint_grads])
        
        # Compute proxy-original discrepancy
        constraint_discrepancy = np.abs(proxy_viols - original_viols)
        
        # Update penalty parameters based on discrepancy
        for i in range(n_constraints):
            if constraint_discrepancy[i] > tolerance:
                penalty_params[i] = min(max_penalty, 
                                      penalty_params[i] * penalty_update_rate)
        
        # Compute Lagrangian gradient for primal update
        lagrangian_grad = obj_grad.copy()
        for i in range(n_constraints):
            # Add penalty for proxy constraint violation
            lagrangian_grad += lambdas[i] * proxy_grads[i]
            # Add penalty for proxy-original discrepancy
            lagrangian_grad += penalty_params[i] * constraint_discrepancy[i] * proxy_grads[i]
        
        # Primal update (gradient descent)
        x_new = x - learning_rate_primal * lagrangian_grad
        
        # Dual update based on regret type
        if regret_type == "external":
            # External regret minimization for dual variables
            dual_grad = proxy_viols - np.array(constraint_bounds)
            lambdas_new = np.maximum(0, lambdas + learning_rate_dual * dual_grad)
        elif regret_type == "swap":
            # Swap regret minimization (more complex update)
            dual_grad = original_viols - np.array(constraint_bounds)
            lambdas_new = np.maximum(0, lambdas + learning_rate_dual * dual_grad)
        else:
            raise ValueError("regret_type must be 'external' or 'swap'")
        
        # Compute regret for convergence assessment
        primal_regret += np.dot(lagrangian_grad, x_new - x)
        dual_regret += np.dot(dual_grad, lambdas_new - lambdas)
        
        # Update variables
        x = x_new
        lambdas = lambdas_new
        
        # Store history
        history['objective'].append(obj_val)
        history['constraint_violations'].append(original_viols.copy())
        history['proxy_violations'].append(proxy_viols.copy())
        history['regret'].append([primal_regret, dual_regret])
        history['penalty_params'].append(penalty_params.copy())
        
        # Check convergence
        if iteration > 10:  # Allow some iterations before checking convergence
            obj_change = abs(history['objective'][-1] - history['objective'][-10])
            constraint_change = np.max(np.abs(
                np.array(history['constraint_violations'][-1]) - 
                np.array(history['constraint_violations'][-10])
            ))
            
            if obj_change < tolerance and constraint_change < tolerance:
                converged = True
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration}: Objective = {obj_val:.6f}, "
                  f"Max constraint violation = {np.max(original_viols):.6f}")
    
    # Final evaluation
    final_objective = objective_func(x)
    final_constraint_violations = np.array([c(x) for c in original_constraints])
    
    return {
        'optimal_x': x,
        'optimal_objective': final_objective,
        'constraint_violations': final_constraint_violations,
        'lagrange_multipliers': lambdas,
        'penalty_parameters': penalty_params,
        'convergence_history': history,
        'converged': converged,
        'iterations': iteration + 1,
        'final_regret': [primal_regret, dual_regret],
        'proxy_original_discrepancy': constraint_discrepancy
    }


if __name__ == "__main__":
    # Example: Constrained quadratic optimization with fairness constraint
    np.random.seed(42)
    
    # Generate synthetic data for fairness-constrained optimization
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.1
    
    # Sensitive attribute (binary)
    sensitive_attr = np.random.binomial(1, 0.5, n_samples)
    
    # Define objective function (mean squared error)
    def objective(w):
        predictions = X @ w
        return np.mean((predictions - y) ** 2)
    
    def objective_grad(w):
        predictions = X @ w
        return 2 * X.T @ (predictions - y) / n_samples
    
    # Define fairness constraint (demographic parity)
    # Original constraint: |P(Y=1|S=0) - P(Y=1|S=1)| <= epsilon
    def original_fairness_constraint(w):
        predictions = X @ w
        pred_binary = (predictions > 0).astype(int)
        
        # Demographic parity violation
        prob_s0 = np.mean(pred_binary[sensitive_attr == 0])
        prob_s1 = np.mean(pred_binary[sensitive_attr == 1])
        return abs(prob_s0 - prob_s1) - 0.1  # epsilon = 0.1
    
    # Proxy constraint (differentiable approximation using sigmoid)
    def proxy_fairness_constraint(w):
        predictions = X @ w
        pred_prob = 1 / (1 + np.exp