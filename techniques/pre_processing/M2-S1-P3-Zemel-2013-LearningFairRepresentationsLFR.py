import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, Tuple
from scipy.optimize import minimize
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

def learning_fair_representations(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    sensitive_attr: Union[np.ndarray, pd.Series],
    n_prototypes: int = 5,
    Ax: float = 0.01,
    Az: float = 50.0,
    Ay: float = 1.0,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    Learning Fair Representations (LFR) implementation.
    
    This technique learns a fair representation Z that maximizes mutual information
    with the target variable Y while constraining mutual information with the
    sensitive attribute A. The optimization problem is:
    max I(Y; Z) subject to I(A; Z) ≤ ε
    
    The method uses a probabilistic mapping to k prototypes and optimizes three
    objectives: reconstruction quality (Ax), fairness constraint (Az), and
    prediction accuracy (Ay).
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features
    y : array-like of shape (n_samples,)
        Target variable
    sensitive_attr : array-like of shape (n_samples,)
        Sensitive attribute (e.g., race, gender)
    n_prototypes : int, default=5
        Number of prototypes in the learned representation
    Ax : float, default=0.01
        Weight for reconstruction quality term
    Az : float, default=50.0
        Weight for fairness constraint (higher values enforce more fairness)
    Ay : float, default=1.0
        Weight for prediction accuracy term
    max_iter : int, default=1000
        Maximum number of optimization iterations
    random_state : int, optional
        Random seed for reproducibility
    test_size : float, default=0.2
        Proportion of data to use for testing
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'prototypes': Learned prototype vectors
        - 'mapping_weights': Weights for mapping to prototypes
        - 'mutual_info_y_z': Mutual information between Y and Z
        - 'mutual_info_a_z': Mutual information between A and Z
        - 'fairness_violation': Amount by which fairness constraint is violated
        - 'reconstruction_error': Mean squared reconstruction error
        - 'prediction_accuracy': Accuracy on test set using fair representation
        - 'original_accuracy': Accuracy on test set using original features
        - 'parameters': Dictionary of input parameters
        
    References
    ----------
    Zemel, R., Wu, Y., Swersky, K., Pitassi, T., & Dwork, C. (2013). 
    Learning fair representations. In International Conference on Machine Learning.
    """
    
    # Input validation
    if not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise TypeError("X must be numpy array or pandas DataFrame")
    if not isinstance(y, (np.ndarray, pd.Series)):
        raise TypeError("y must be numpy array or pandas Series")
    if not isinstance(sensitive_attr, (np.ndarray, pd.Series)):
        raise TypeError("sensitive_attr must be numpy array or pandas Series")
    
    # Convert to numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(sensitive_attr, pd.Series):
        sensitive_attr = sensitive_attr.values
    
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    sensitive_attr = np.asarray(sensitive_attr)
    
    if X.shape[0] != len(y) or X.shape[0] != len(sensitive_attr):
        raise ValueError("X, y, and sensitive_attr must have same number of samples")
    
    if n_prototypes <= 0:
        raise ValueError("n_prototypes must be positive")
    
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1")
    
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Encode categorical variables
    le_y = LabelEncoder()
    le_a = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    a_encoded = le_a.fit_transform(sensitive_attr)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(
        X_scaled, y_encoded, a_encoded, test_size=test_size, 
        random_state=random_state, stratify=y_encoded
    )
    
    n_samples, n_features = X_train.shape
    n_classes_y = len(np.unique(y_encoded))
    n_classes_a = len(np.unique(a_encoded))
    
    # Initialize parameters
    # Prototypes: k x d matrix
    prototypes = np.random.randn(n_prototypes, n_features) * 0.1
    
    # Mapping weights: n x k matrix (probability of each sample belonging to each prototype)
    w = np.random.rand(n_samples, n_prototypes)
    w = w / w.sum(axis=1, keepdims=True)  # Normalize to probabilities
    
    def compute_mutual_information(X: np.ndarray, Y: np.ndarray, bins: int = 10) -> float:
        """Compute mutual information between continuous X and discrete Y"""
        if len(np.unique(Y)) == 1:
            return 0.0
        
        # Discretize continuous variables
        if X.ndim == 1:
            X_discrete = np.digitize(X, np.histogram(X, bins=bins)[1][:-1])
        else:
            # For multivariate X, use first principal component
            X_discrete = np.digitize(X.mean(axis=1), 
                                   np.histogram(X.mean(axis=1), bins=bins)[1][:-1])
        
        # Compute joint and marginal probabilities
        joint_counts = np.histogram2d(X_discrete, Y, 
                                    bins=[len(np.unique(X_discrete)), len(np.unique(Y))])[0]
        joint_probs = joint_counts / joint_counts.sum()
        
        # Marginal probabilities
        px = joint_probs.sum(axis=1)
        py = joint_probs.sum(axis=0)
        
        # Mutual information
        mi = 0.0
        for i in range(len(px)):
            for j in range(len(py)):
                if joint_probs[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint_probs[i, j] * np.log(joint_probs[i, j] / (px[i] * py[j]))
        
        return max(0.0, mi)
    
    def objective_function(params: np.ndarray) -> float:
        """Objective function for LFR optimization"""
        # Unpack parameters
        prototypes_flat = params[:n_prototypes * n_features]
        w_flat = params[n_prototypes * n_features:]
        
        prototypes_curr = prototypes_flat.reshape(n_prototypes, n_features)
        w_curr = w_flat.reshape(n_samples, n_prototypes)
        
        # Ensure w represents valid probabilities
        w_curr = np.abs(w_curr)
        w_curr = w_curr / (w_curr.sum(axis=1, keepdims=True) + 1e-8)
        
        # Compute representation Z as weighted combination of prototypes
        Z = w_curr @ prototypes_curr
        
        # Reconstruction error (Lx term)
        reconstruction_error = np.mean((X_train - Z) ** 2)
        
        # Fairness constraint (Lz term) - mutual information between A and Z
        mi_a_z = compute_mutual_information(Z, a_train)
        
        # Prediction accuracy (Ly term) - mutual information between Y and Z
        mi_y_z = compute_mutual_information(Z, y_train)
        
        # Combined objective (minimize negative of desired objective)
        objective = Ax * reconstruction_error + Az * mi_a_z - Ay * mi_y_z
        
        return objective
    
    # Pack initial parameters
    initial_params = np.concatenate([prototypes.flatten(), w.flatten()])
    
    # Optimize
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            objective_function,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'disp': False}
        )
    
    # Unpack optimized parameters
    prototypes_opt = result.x[:n_prototypes * n_features].reshape(n_prototypes, n_features)
    w_opt = result.x[n_prototypes * n_features:].reshape(n_samples, n_prototypes)
    w_opt = np.abs(w_opt)
    w_opt = w_opt / (w_opt.sum(axis=1, keepdims=True) + 1e-8)
    
    # Compute final representation for training data
    Z_train = w_opt @ prototypes_opt
    
    # For test data, compute mapping weights
    def compute_test_representation(X_test_data: np.ndarray) -> np.ndarray:
        """Compute representation for test data"""
        n_test = X_test_data.shape[0]
        w_test = np.zeros((n_test, n_prototypes))
        
        for i in range(n_test):
            # Compute distances to prototypes
            distances = np.sum((prototypes_opt - X_test_data[i]) ** 2, axis=1)
            # Convert to probabilities (softmax)
            w_test[i] = np.exp(-distances) / np.sum(np.exp(-distances))
        
        return w_test @ prototypes_opt
    
    Z_test = compute_test_representation(X_test)
    
    # Compute final metrics
    mi_y_z_final = compute_mutual_information(Z_train, y_train)
    mi_a_z_final = compute_mutual_information(Z_train, a_train)
    reconstruction_error_final = np.mean((X_train - Z_train) ** 2)
    
    # Evaluate prediction accuracy using fair representation
    clf_fair = LogisticRegression(random_state=random_state, max_iter=1000)
    clf_fair.fit(Z_train, y_train)
    y_pred_fair = clf_fair.predict(Z_test)
    accuracy_fair = accuracy_score(y_test, y_pred_fair)
    
    # Evaluate prediction accuracy using original features
    clf_orig = LogisticRegression(random_state=random_state, max_iter=1000)
    clf_orig.fit(X_train, y_train)
    y_pred_