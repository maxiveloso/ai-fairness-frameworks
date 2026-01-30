import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy import optimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings

def counterfactually_fair_representations_learning(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    sensitive_attrs: Union[List[str], List[int]],
    causal_graph: Dict[str, List[str]],
    feature_names: Optional[List[str]] = None,
    n_components: int = 10,
    lambda_penalty: float = 1.0,
    max_iter: int = 1000,
    learning_rate: float = 0.001,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Implement Counterfactually Fair Representations Learning algorithm.
    
    This algorithm learns fair representations by minimizing the difference in outcomes
    across counterfactual worlds where sensitive attributes are intervened upon.
    The three-step process involves: 1) Abduction - inferring latent noise variables,
    2) Action - performing do-interventions on sensitive attributes, 3) Prediction -
    computing counterfactual outcomes.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features
    y : array-like of shape (n_samples,)
        Target variable
    sensitive_attrs : list
        List of sensitive attribute column names or indices
    causal_graph : dict
        Dictionary representing causal DAG structure where keys are nodes
        and values are lists of parent nodes
    feature_names : list, optional
        Names of features if X is numpy array
    n_components : int, default=10
        Number of components in learned representation
    lambda_penalty : float, default=1.0
        Penalty weight for counterfactual fairness constraint
    max_iter : int, default=1000
        Maximum number of optimization iterations
    learning_rate : float, default=0.001
        Learning rate for optimization
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print optimization progress
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'fair_representations': Learned fair representations
        - 'counterfactual_fairness_score': Aggregate fairness measure
        - 'individual_fairness_scores': Per-sample fairness scores
        - 'reconstruction_error': Quality of representation
        - 'causal_effects': Estimated causal effects
        - 'optimization_history': Training loss history
    """
    
    # Input validation
    if not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise TypeError("X must be numpy array or pandas DataFrame")
    if not isinstance(y, (np.ndarray, pd.Series)):
        raise TypeError("y must be numpy array or pandas Series")
    
    # Convert to numpy arrays and standardize format
    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or list(X.columns)
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
        
    n_samples, n_features = X.shape
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Validate sensitive attributes
    if isinstance(sensitive_attrs[0], str):
        sensitive_indices = [feature_names.index(attr) for attr in sensitive_attrs]
    else:
        sensitive_indices = sensitive_attrs
        
    if not all(0 <= idx < n_features for idx in sensitive_indices):
        raise ValueError("Invalid sensitive attribute indices")
    
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize representation learning components
    class CounterfactuallyFairEncoder:
        def __init__(self, input_dim, repr_dim, sensitive_indices):
            self.input_dim = input_dim
            self.repr_dim = repr_dim
            self.sensitive_indices = sensitive_indices
            
            # Initialize encoder and decoder weights
            self.W_enc = np.random.normal(0, 0.1, (input_dim, repr_dim))
            self.b_enc = np.zeros(repr_dim)
            self.W_dec = np.random.normal(0, 0.1, (repr_dim, input_dim))
            self.b_dec = np.zeros(input_dim)
            
            # Predictor weights
            self.W_pred = np.random.normal(0, 0.1, repr_dim)
            self.b_pred = 0.0
            
        def encode(self, X):
            """Encode input to representation space"""
            return np.tanh(X @ self.W_enc + self.b_enc)
        
        def decode(self, Z):
            """Decode representation back to input space"""
            return Z @ self.W_dec + self.b_dec
        
        def predict(self, Z):
            """Predict outcome from representation"""
            return Z @ self.W_pred + self.b_pred
        
        def get_params(self):
            """Get all parameters as flat array"""
            return np.concatenate([
                self.W_enc.flatten(), self.b_enc,
                self.W_dec.flatten(), self.b_dec,
                self.W_pred, [self.b_pred]
            ])
        
        def set_params(self, params):
            """Set parameters from flat array"""
            idx = 0
            
            # Encoder weights
            w_enc_size = self.input_dim * self.repr_dim
            self.W_enc = params[idx:idx+w_enc_size].reshape(self.input_dim, self.repr_dim)
            idx += w_enc_size
            
            self.b_enc = params[idx:idx+self.repr_dim]
            idx += self.repr_dim
            
            # Decoder weights
            w_dec_size = self.repr_dim * self.input_dim
            self.W_dec = params[idx:idx+w_dec_size].reshape(self.repr_dim, self.input_dim)
            idx += w_dec_size
            
            self.b_dec = params[idx:idx+self.input_dim]
            idx += self.input_dim
            
            # Predictor weights
            self.W_pred = params[idx:idx+self.repr_dim]
            idx += self.repr_dim
            
            self.b_pred = params[idx]
    
    # Initialize encoder
    encoder = CounterfactuallyFairEncoder(n_features, n_components, sensitive_indices)
    
    def compute_counterfactual_outcomes(X, encoder, sensitive_indices):
        """
        Compute counterfactual outcomes by intervening on sensitive attributes.
        
        This implements the three-step counterfactual inference:
        1. Abduction: Infer noise variables from observed data
        2. Action: Intervene on sensitive attributes
        3. Prediction: Compute counterfactual outcomes
        """
        Z = encoder.encode(X)  # Step 1: Abduction (encode to latent space)
        
        counterfactuals = []
        
        # Generate counterfactuals by flipping sensitive attributes
        for flip_combination in range(2**len(sensitive_indices)):
            X_cf = X.copy()
            
            # Step 2: Action (intervene on sensitive attributes)
            for i, sens_idx in enumerate(sensitive_indices):
                if (flip_combination >> i) & 1:
                    # Flip binary sensitive attribute or negate continuous
                    unique_vals = np.unique(X[:, sens_idx])
                    if len(unique_vals) == 2:
                        X_cf[:, sens_idx] = 1 - X_cf[:, sens_idx]
                    else:
                        X_cf[:, sens_idx] = -X_cf[:, sens_idx]
            
            # Step 3: Prediction (compute counterfactual outcomes)
            Z_cf = encoder.encode(X_cf)
            y_cf = encoder.predict(Z_cf)
            counterfactuals.append(y_cf)
        
        return counterfactuals
    
    def objective_function(params):
        """
        Objective function combining reconstruction loss and fairness penalty.
        
        The fairness penalty measures the average difference in outcomes
        across counterfactual worlds where sensitive attributes are intervened.
        """
        encoder.set_params(params)
        
        # Reconstruction loss
        Z = encoder.encode(X_scaled)
        X_recon = encoder.decode(Z)
        reconstruction_loss = np.mean((X_scaled - X_recon)**2)
        
        # Prediction loss
        y_pred = encoder.predict(Z)
        prediction_loss = np.mean((y - y_pred)**2)
        
        # Counterfactual fairness penalty
        counterfactuals = compute_counterfactual_outcomes(X_scaled, encoder, sensitive_indices)
        
        fairness_penalty = 0.0
        n_counterfactuals = len(counterfactuals)
        
        # Compute pairwise differences between counterfactual outcomes
        for i in range(n_counterfactuals):
            for j in range(i+1, n_counterfactuals):
                fairness_penalty += np.mean((counterfactuals[i] - counterfactuals[j])**2)
        
        if n_counterfactuals > 1:
            fairness_penalty /= (n_counterfactuals * (n_counterfactuals - 1) / 2)
        
        total_loss = reconstruction_loss + prediction_loss + lambda_penalty * fairness_penalty
        
        return total_loss
    
    # Optimization
    initial_params = encoder.get_params()
    optimization_history = []
    
    def callback(params):
        if verbose and len(optimization_history) % 100 == 0:
            print(f"Iteration {len(optimization_history)}: Loss = {optimization_history[-1]:.6f}")
    
    def objective_with_history(params):
        loss = objective_function(params)
        optimization_history.append(loss)
        return loss
    
    # Run optimization
    try:
        result = optimize.minimize(
            objective_with_history,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'disp': verbose}
        )
        
        # Set final parameters
        encoder.set_params(result.x)
        
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}. Using initial parameters.")
    
    # Compute final results
    Z_final = encoder.encode(X_scaled)
    X_recon_final = encoder.decode(Z_final)
    y_pred_final = encoder.predict(Z_final)
    
    # Compute counterfactual fairness metrics
    counterfactuals_final = compute_counterfactual_outcomes(X_scaled, encoder, sensitive_indices)
    
    # Individual fairness scores (variance across counterfactuals for each sample)
    individual_fairness_scores = np.zeros(n_samples)
    if len(counterfactuals_final) > 1:
        cf_matrix = np.array(counterfactuals_final).T  # Shape: (n_samples, n_counterfactuals)
        individual_fairness_scores = np.var(cf_matrix, axis=1)
    
    # Aggregate fairness score
    counterfactual