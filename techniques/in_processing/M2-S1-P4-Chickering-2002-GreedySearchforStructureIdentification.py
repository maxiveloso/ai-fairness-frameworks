import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy import stats
from itertools import combinations
import warnings

def greedy_search_structure_identification(
    data: Union[np.ndarray, pd.DataFrame],
    score_function: str = 'bic',
    max_parents: Optional[int] = None,
    alpha: float = 0.05,
    max_iterations: int = 100,
    random_state: Optional[int] = None
) -> Dict:
    """
    Perform greedy search for structure identification in directed acyclic graphs (DAGs).
    
    This implementation follows Chickering's (2002) three-phase greedy search algorithm:
    1. Forward phase: Add edges that improve the score
    2. Backward phase: Remove edges that improve the score  
    3. Edge flipping phase: Reorient edges that improve the score
    
    The algorithm searches through equivalence classes of DAGs using local scoring
    functions to identify optimal causal structure from observational data.
    
    Parameters
    ----------
    data : array-like or DataFrame of shape (n_samples, n_features)
        Input data matrix where rows are observations and columns are variables
    score_function : str, default='bic'
        Scoring function to optimize ('bic', 'aic', or 'likelihood')
    max_parents : int, optional
        Maximum number of parents allowed for any node (default: n_features - 1)
    alpha : float, default=0.05
        Significance level for statistical tests
    max_iterations : int, default=100
        Maximum number of iterations for the search algorithm
    random_state : int, optional
        Random seed for reproducible results
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'adjacency_matrix': Final adjacency matrix of the learned DAG
        - 'score': Final score of the learned structure
        - 'edges': List of directed edges (parent, child) tuples
        - 'iterations': Number of iterations performed
        - 'phase_scores': Scores after each phase
        - 'convergence': Whether algorithm converged
        - 'node_names': Variable names if DataFrame input
    """
    
    # Input validation
    if isinstance(data, pd.DataFrame):
        node_names = list(data.columns)
        data_array = data.values
    else:
        data_array = np.asarray(data)
        node_names = [f'X{i}' for i in range(data_array.shape[1])]
    
    if data_array.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    
    n_samples, n_features = data_array.shape
    
    if n_samples < n_features:
        warnings.warn("Number of samples is less than number of features")
    
    if max_parents is None:
        max_parents = n_features - 1
    
    if max_parents >= n_features:
        max_parents = n_features - 1
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize adjacency matrix (0 = no edge, 1 = edge from i to j)
    adjacency_matrix = np.zeros((n_features, n_features), dtype=int)
    
    # Define scoring function
    def calculate_score(adj_matrix: np.ndarray) -> float:
        """Calculate BIC/AIC score for given adjacency matrix structure"""
        total_score = 0.0
        
        for j in range(n_features):
            # Find parents of node j
            parents = np.where(adj_matrix[:, j] == 1)[0]
            
            if len(parents) == 0:
                # No parents - just fit intercept
                y = data_array[:, j]
                mse = np.var(y)
                log_likelihood = -0.5 * n_samples * np.log(2 * np.pi * mse) - 0.5 * n_samples
                n_params = 1  # intercept only
            else:
                # Linear regression with parents as predictors
                X = data_array[:, parents]
                y = data_array[:, j]
                
                # Add intercept
                X_with_intercept = np.column_stack([np.ones(n_samples), X])
                
                try:
                    # Ordinary least squares
                    beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                    y_pred = X_with_intercept @ beta
                    residuals = y - y_pred
                    mse = np.mean(residuals**2)
                    
                    if mse <= 0:
                        mse = 1e-10  # Avoid log(0)
                    
                    log_likelihood = -0.5 * n_samples * np.log(2 * np.pi * mse) - 0.5 * n_samples
                    n_params = len(parents) + 1  # coefficients + intercept
                    
                except np.linalg.LinAlgError:
                    # Singular matrix - penalize heavily
                    log_likelihood = -np.inf
                    n_params = len(parents) + 1
            
            # Apply penalty based on score function
            if score_function.lower() == 'bic':
                penalty = 0.5 * n_params * np.log(n_samples)
            elif score_function.lower() == 'aic':
                penalty = n_params
            else:  # likelihood
                penalty = 0
            
            node_score = log_likelihood - penalty
            total_score += node_score
        
        return total_score
    
    def is_acyclic(adj_matrix: np.ndarray) -> bool:
        """Check if adjacency matrix represents a DAG (no cycles)"""
        # Use topological sort approach
        in_degree = np.sum(adj_matrix, axis=0)
        queue = [i for i in range(n_features) if in_degree[i] == 0]
        visited = 0
        
        while queue:
            node = queue.pop(0)
            visited += 1
            
            # Remove edges from this node
            for neighbor in range(n_features):
                if adj_matrix[node, neighbor] == 1:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        return visited == n_features
    
    # Initialize tracking variables
    current_score = calculate_score(adjacency_matrix)
    phase_scores = [current_score]
    iteration = 0
    converged = False
    
    for iteration in range(max_iterations):
        improved = False
        best_score = current_score
        best_matrix = adjacency_matrix.copy()
        
        # Phase 1: Forward phase - add edges
        for i in range(n_features):
            for j in range(n_features):
                if i != j and adjacency_matrix[i, j] == 0:
                    # Check if adding edge would violate constraints
                    current_parents = np.sum(adjacency_matrix[:, j])
                    if current_parents >= max_parents:
                        continue
                    
                    # Try adding edge i -> j
                    test_matrix = adjacency_matrix.copy()
                    test_matrix[i, j] = 1
                    
                    # Check if still acyclic
                    if is_acyclic(test_matrix):
                        test_score = calculate_score(test_matrix)
                        if test_score > best_score:
                            best_score = test_score
                            best_matrix = test_matrix
                            improved = True
        
        # Phase 2: Backward phase - remove edges
        for i in range(n_features):
            for j in range(n_features):
                if adjacency_matrix[i, j] == 1:
                    # Try removing edge i -> j
                    test_matrix = adjacency_matrix.copy()
                    test_matrix[i, j] = 0
                    
                    test_score = calculate_score(test_matrix)
                    if test_score > best_score:
                        best_score = test_score
                        best_matrix = test_matrix
                        improved = True
        
        # Phase 3: Edge flipping phase - reorient edges
        for i in range(n_features):
            for j in range(n_features):
                if adjacency_matrix[i, j] == 1:
                    # Check if we can flip edge i -> j to j -> i
                    current_parents_i = np.sum(adjacency_matrix[:, i])
                    if current_parents_i >= max_parents:
                        continue
                    
                    # Try flipping edge
                    test_matrix = adjacency_matrix.copy()
                    test_matrix[i, j] = 0
                    test_matrix[j, i] = 1
                    
                    # Check if still acyclic
                    if is_acyclic(test_matrix):
                        test_score = calculate_score(test_matrix)
                        if test_score > best_score:
                            best_score = test_score
                            best_matrix = test_matrix
                            improved = True
        
        # Update adjacency matrix and score
        adjacency_matrix = best_matrix
        current_score = best_score
        phase_scores.append(current_score)
        
        # Check for convergence
        if not improved:
            converged = True
            break
    
    # Extract edges from final adjacency matrix
    edges = []
    for i in range(n_features):
        for j in range(n_features):
            if adjacency_matrix[i, j] == 1:
                edges.append((node_names[i], node_names[j]))
    
    return {
        'adjacency_matrix': adjacency_matrix,
        'score': current_score,
        'edges': edges,
        'iterations': iteration + 1,
        'phase_scores': phase_scores,
        'convergence': converged,
        'node_names': node_names
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic data with known causal structure
    # True structure: X0 -> X1 -> X2, X0 -> X2
    n_samples = 200
    
    # Generate data following the causal model
    X0 = np.random.normal(0, 1, n_samples)
    X1 = 0.8 * X0 + np.random.normal(0, 0.5, n_samples)
    X2 = 0.6 * X0 + 0.7 * X1 + np.random.normal(0, 0.3, n_samples)
    
    data = pd.DataFrame({
        'X0': X0,
        'X1': X1, 
        'X2': X2
    })
    
    print("Synthetic Data Shape:", data.shape)
    print("\nTrue causal structure: X0 -> X1 -> X2, X0 -> X2")
    
    # Apply greedy search structure identification
    result = greedy_search_structure_identification(
        data=data,
        score_function='bic',
        max_parents=2,
        random_state=42
    )
    
    print(f"\nGreedy Search Results:")
    print(f"Final Score: {result['score']:.3f}")
    print(f"Iterations: {result['iterations']}")
    print(f"Converged: {result['convergence']}")
    
    print(f"\nLearned Adjacency Matrix:")
    print(result['adjacency_matrix'])
    
    print(f"\nLearned Edges:")
    for edge in result['edges']:
        print(f"  {edge[0]} -> {edge