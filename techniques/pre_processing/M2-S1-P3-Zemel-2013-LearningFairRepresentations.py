import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
from scipy.optimize import minimize
from typing import Union, Dict, Any, Optional, Tuple
import warnings

def learning_fair_representations(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    protected_attribute: Union[np.ndarray, pd.Series],
    k: int = 5,
    Ax: float = 1.0,
    Ay: float = 1.0,
    Az: float = 1.0,
    epsilon: float = 0.1,
    max_iter: int = 100,
    random_state: Optional[int] = None,
    standardize: bool = True
) -> Dict[str, Any]:
    """
    Learn Fair Representations using prototype-based approach.
    
    This method learns a representation Z = g(X) that maximizes mutual information
    with the target variable Y while constraining mutual information with the
    protected attribute A to be below epsilon. Uses a prototype-based approach
    where representations are probabilistic assignments to k prototypes.
    
    The optimization problem is:
    max I(Y; Z) subject to I(A; Z) ≤ ε
    
    where the objective balances three terms:
    - Reconstruction loss: how well Z reconstructs X
    - Prediction loss: how well Z predicts Y  
    - Fairness constraint: limiting information about protected attribute A
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features
    y : array-like of shape (n_samples,)
        Target variable
    protected_attribute : array-like of shape (n_samples,)
        Protected attribute (e.g., race, gender)
    k : int, default=5
        Number of prototypes in the representation
    Ax : float, default=1.0
        Weight for reconstruction loss term
    Ay : float, default=1.0
        Weight for prediction loss term
    Az : float, default=1.0
        Weight for fairness constraint term
    epsilon : float, default=0.1
        Maximum allowed mutual information between representation and protected attribute
    max_iter : int, default=100
        Maximum number of optimization iterations
    random_state : int, optional
        Random seed for reproducibility
    standardize : bool, default=True
        Whether to standardize input features
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'prototypes': Learned prototype vectors
        - 'representation': Fair representation Z for input data
        - 'mutual_info_y_z': Mutual information between Y and Z
        - 'mutual_info_a_z': Mutual information between A and Z
        - 'fairness_satisfied': Whether fairness constraint is satisfied
        - 'reconstruction_loss': Final reconstruction loss
        - 'prediction_loss': Final prediction loss
        - 'total_loss': Final total objective value
        - 'converged': Whether optimization converged
        
    References
    ----------
    Zemel, R., Wu, Y., Swersky, K., Pitassi, T., & Dwork, C. (2013). 
    Learning fair representations. In International Conference on Machine Learning (pp. 325–333).
    """
    
    # Input validation
    X = np.asarray(X)
    y = np.asarray(y)
    protected_attribute = np.asarray(protected_attribute)
    
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if y.ndim != 1:
        raise ValueError("y must be 1-dimensional")
    if protected_attribute.ndim != 1:
        raise ValueError("protected_attribute must be 1-dimensional")
    if len(X) != len(y) or len(X) != len(protected_attribute):
        raise ValueError("X, y, and protected_attribute must have same number of samples")
    if k <= 0:
        raise ValueError("k must be positive")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if Ax < 0 or Ay < 0 or Az < 0:
        raise ValueError("All weight parameters must be non-negative")
    
    n_samples, n_features = X.shape
    
    # Standardize features if requested
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.copy()
        scaler = None
    
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize prototypes using k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans.fit(X_scaled)
    prototypes_init = kmeans.cluster_centers_
    
    # Convert categorical variables to numeric if needed
    y_numeric = pd.Categorical(y).codes if y.dtype == 'object' else y
    a_numeric = pd.Categorical(protected_attribute).codes if protected_attribute.dtype == 'object' else protected_attribute
    
    def softmax(x, axis=None):
        """Compute softmax values"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def compute_representation(X_data, prototypes):
        """Compute probabilistic representation Z given prototypes"""
        # Compute distances to prototypes
        distances = np.zeros((len(X_data), k))
        for i, prototype in enumerate(prototypes):
            distances[:, i] = np.sum((X_data - prototype) ** 2, axis=1)
        
        # Convert to probabilities using softmax (negative distances)
        Z = softmax(-distances, axis=1)
        return Z
    
    def discretize_for_mi(arr, bins=10):
        """Discretize continuous array for mutual information calculation"""
        if len(np.unique(arr)) <= bins:
            return arr
        return pd.cut(arr, bins=bins, labels=False, duplicates='drop')
    
    def compute_mutual_information(x, y):
        """Compute mutual information between two variables"""
        # Discretize if continuous
        x_disc = discretize_for_mi(x.flatten() if x.ndim > 1 else x)
        y_disc = discretize_for_mi(y.flatten() if y.ndim > 1 else y)
        
        # Handle NaN values
        valid_mask = ~(pd.isna(x_disc) | pd.isna(y_disc))
        if np.sum(valid_mask) == 0:
            return 0.0
            
        return mutual_info_score(x_disc[valid_mask], y_disc[valid_mask])
    
    def objective_function(params):
        """Objective function to minimize (negative of what we want to maximize)"""
        prototypes = params.reshape(k, n_features)
        
        # Compute representation
        Z = compute_representation(X_scaled, prototypes)
        
        # Reconstruction loss: how well can we reconstruct X from Z
        X_reconstructed = Z @ prototypes
        reconstruction_loss = np.mean(np.sum((X_scaled - X_reconstructed) ** 2, axis=1))
        
        # Prediction loss: how well can Z predict y
        # Use correlation as proxy for predictive power
        Z_flat = Z.flatten()
        y_repeated = np.repeat(y_numeric, k)
        prediction_loss = -np.abs(np.corrcoef(Z_flat, y_repeated)[0, 1])
        if np.isnan(prediction_loss):
            prediction_loss = 0
        
        # Fairness constraint: mutual information between Z and protected attribute
        mi_a_z = 0
        for j in range(k):
            mi_a_z += compute_mutual_information(Z[:, j], a_numeric)
        mi_a_z /= k  # Average across prototype dimensions
        
        # Penalty for violating fairness constraint
        fairness_penalty = max(0, mi_a_z - epsilon) ** 2
        
        # Total objective (we minimize, so negate terms we want to maximize)
        total_loss = (Ax * reconstruction_loss + 
                     Ay * prediction_loss + 
                     Az * fairness_penalty)
        
        return total_loss
    
    # Optimize prototypes
    initial_params = prototypes_init.flatten()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            objective_function,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'disp': False}
        )
    
    # Extract final results
    final_prototypes = result.x.reshape(k, n_features)
    final_Z = compute_representation(X_scaled, final_prototypes)
    
    # Compute final metrics
    X_reconstructed = final_Z @ final_prototypes
    reconstruction_loss = np.mean(np.sum((X_scaled - X_reconstructed) ** 2, axis=1))
    
    # Mutual information between Y and Z
    mi_y_z = 0
    for j in range(k):
        mi_y_z += compute_mutual_information(final_Z[:, j], y_numeric)
    mi_y_z /= k
    
    # Mutual information between A and Z
    mi_a_z = 0
    for j in range(k):
        mi_a_z += compute_mutual_information(final_Z[:, j], a_numeric)
    mi_a_z /= k
    
    # Prediction loss
    Z_flat = final_Z.flatten()
    y_repeated = np.repeat(y_numeric, k)
    prediction_corr = np.corrcoef(Z_flat, y_repeated)[0, 1]
    if np.isnan(prediction_corr):
        prediction_corr = 0
    prediction_loss = -np.abs(prediction_corr)
    
    return {
        'prototypes': final_prototypes,
        'representation': final_Z,
        'mutual_info_y_z': mi_y_z,
        'mutual_info_a_z': mi_a_z,
        'fairness_satisfied': mi_a_z <= epsilon,
        'reconstruction_loss': reconstruction_loss,
        'prediction_loss': prediction_loss,
        'total_loss': result.fun,
        'converged': result.success,
        'scaler': scaler
    }

if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic dataset
    n_samples = 500
    n_features = 4
    
    # Protected attribute (binary: 0 or 1)
    protected_attr = np.random.binomial(1, 0.5, n_samples)
    
    # Features correlated with protected attribute
    X = np.random.randn(n_samples, n_features)
    X[:, 0] += protected_attr * 2  # First feature correlated with protected attribute
    
    # Target variable influenced by features and protected attribute
    y = (X[:, 0] + X[:, 1] + 0.5 * protected_attr + 
         np.random.randn(n_samples) * 0.1)
    
    print("Learning Fair Representations Example")
    print("=" * 50)
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Protected attribute distribution: {np.bincount(protected_attr)}")
    
    # Apply Learning Fair Representations
    result = learning