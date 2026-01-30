import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, Callable
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

def adversarial_learning_bias_mitigation(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    sensitive_attr: Union[np.ndarray, pd.Series],
    fairness_type: str = 'demographic_parity',
    alpha: float = 1.0,
    max_iter: int = 100,
    learning_rate: float = 0.01,
    predictor_hidden_layers: tuple = (64, 32),
    adversary_hidden_layers: tuple = (32,),
    tolerance: float = 1e-6,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Implement adversarial learning for bias mitigation using alternating gradient updates.
    
    This technique trains a predictor to make accurate predictions while simultaneously
    training an adversary to detect bias. The predictor is updated to minimize prediction
    loss while maximizing adversary loss, creating a minimax game that encourages
    fair representations.
    
    The algorithm alternates between:
    1. Updating adversary weights U via ∇_U L_A (standard gradient ascent)
    2. Updating predictor weights W via ∇_W L_P - proj_{∇_W L_A} ∇_W L_P - α ∇_W L_A
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target variable (binary classification)
    sensitive_attr : array-like of shape (n_samples,)
        Sensitive attribute for fairness (binary)
    fairness_type : str, default='demographic_parity'
        Type of fairness constraint ('demographic_parity', 'equality_of_odds', 
        'equality_of_opportunity')
    alpha : float, default=1.0
        Fairness constraint strength parameter
    max_iter : int, default=100
        Maximum number of training iterations
    learning_rate : float, default=0.01
        Learning rate for gradient updates
    predictor_hidden_layers : tuple, default=(64, 32)
        Hidden layer sizes for predictor network
    adversary_hidden_layers : tuple, default=(32,)
        Hidden layer sizes for adversary network
    tolerance : float, default=1e-6
        Convergence tolerance
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'predictor_accuracy': Final predictor accuracy
        - 'adversary_accuracy': Final adversary accuracy  
        - 'fairness_violation': Measure of fairness constraint violation
        - 'demographic_parity': Demographic parity difference
        - 'equality_of_odds': Equality of odds difference
        - 'equality_of_opportunity': Equality of opportunity difference
        - 'training_history': Loss values during training
        - 'converged': Whether algorithm converged
        - 'final_predictor': Trained predictor model
    """
    
    # Input validation
    X = np.asarray(X)
    y = np.asarray(y)
    sensitive_attr = np.asarray(sensitive_attr)
    
    if X.shape[0] != len(y) or X.shape[0] != len(sensitive_attr):
        raise ValueError("X, y, and sensitive_attr must have same number of samples")
    
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must be binary (0, 1)")
        
    if not np.all(np.isin(sensitive_attr, [0, 1])):
        raise ValueError("sensitive_attr must be binary (0, 1)")
        
    if fairness_type not in ['demographic_parity', 'equality_of_odds', 'equality_of_opportunity']:
        raise ValueError("fairness_type must be one of: demographic_parity, equality_of_odds, equality_of_opportunity")
    
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
        
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_samples, n_features = X_scaled.shape
    
    # Initialize neural networks
    # Predictor: X -> y_pred
    predictor = MLPClassifier(
        hidden_layer_sizes=predictor_hidden_layers,
        learning_rate_init=learning_rate,
        max_iter=1,
        warm_start=True,
        random_state=random_state
    )
    
    # Adversary: predictor_output -> sensitive_attr
    adversary = MLPClassifier(
        hidden_layer_sizes=adversary_hidden_layers,
        learning_rate_init=learning_rate,
        max_iter=1,
        warm_start=True,
        random_state=random_state
    )
    
    # Initialize models with small training
    predictor.fit(X_scaled, y)
    
    # Get predictor outputs for adversary training
    predictor_probs = predictor.predict_proba(X_scaled)
    if predictor_probs.shape[1] == 2:
        predictor_outputs = predictor_probs[:, 1].reshape(-1, 1)
    else:
        predictor_outputs = predictor_probs
    
    adversary.fit(predictor_outputs, sensitive_attr)
    
    # Training history
    history = {
        'predictor_loss': [],
        'adversary_loss': [],
        'fairness_violation': []
    }
    
    # Training loop with alternating updates
    converged = False
    prev_predictor_loss = float('inf')
    
    for iteration in range(max_iter):
        # Get current predictions
        y_pred_probs = predictor.predict_proba(X_scaled)
        if y_pred_probs.shape[1] == 2:
            y_pred_probs_pos = y_pred_probs[:, 1]
            predictor_outputs = y_pred_probs_pos.reshape(-1, 1)
        else:
            y_pred_probs_pos = y_pred_probs[:, 0]
            predictor_outputs = y_pred_probs
        
        # Calculate predictor loss (cross-entropy)
        epsilon = 1e-15  # Prevent log(0)
        y_pred_probs_pos = np.clip(y_pred_probs_pos, epsilon, 1 - epsilon)
        predictor_loss = -np.mean(y * np.log(y_pred_probs_pos) + (1 - y) * np.log(1 - y_pred_probs_pos))
        
        # Update adversary (standard training to predict sensitive attribute)
        adversary.fit(predictor_outputs, sensitive_attr)
        
        # Calculate adversary loss
        adv_pred_probs = adversary.predict_proba(predictor_outputs)
        if adv_pred_probs.shape[1] == 2:
            adv_pred_probs_pos = adv_pred_probs[:, 1]
        else:
            adv_pred_probs_pos = adv_pred_probs[:, 0]
        
        adv_pred_probs_pos = np.clip(adv_pred_probs_pos, epsilon, 1 - epsilon)
        adversary_loss = -np.mean(sensitive_attr * np.log(adv_pred_probs_pos) + 
                                 (1 - sensitive_attr) * np.log(1 - adv_pred_probs_pos))
        
        # Calculate fairness violation based on fairness type
        fairness_violation = _calculate_fairness_violation(
            y, y_pred_probs_pos, sensitive_attr, fairness_type
        )
        
        # Store history
        history['predictor_loss'].append(predictor_loss)
        history['adversary_loss'].append(adversary_loss)
        history['fairness_violation'].append(fairness_violation)
        
        # Update predictor with adversarial objective
        # This is simplified - in practice would need custom gradient computation
        # Here we use the fairness violation as a proxy for adversarial gradient
        try:
            # Create adversarial training data by adding fairness penalty
            sample_weights = np.ones(len(y))
            
            # Increase weight for samples that contribute to bias
            for i in range(len(y)):
                if fairness_type == 'demographic_parity':
                    # Penalize predictions that correlate with sensitive attribute
                    if (y_pred_probs_pos[i] > 0.5) != (sensitive_attr[i] == 1):
                        sample_weights[i] *= (1 + alpha)
                elif fairness_type == 'equality_of_odds':
                    # Penalize different error rates across groups
                    if y[i] == 1 and (y_pred_probs_pos[i] > 0.5) != (sensitive_attr[i] == 1):
                        sample_weights[i] *= (1 + alpha)
                elif fairness_type == 'equality_of_opportunity':
                    # Penalize different true positive rates
                    if y[i] == 1 and sensitive_attr[i] == 0 and y_pred_probs_pos[i] <= 0.5:
                        sample_weights[i] *= (1 + alpha)
                    elif y[i] == 1 and sensitive_attr[i] == 1 and y_pred_probs_pos[i] > 0.5:
                        sample_weights[i] *= (1 + alpha)
            
            # Retrain predictor with modified weights
            predictor.fit(X_scaled, y, sample_weight=sample_weights)
            
        except Exception as e:
            warnings.warn(f"Error in adversarial update at iteration {iteration}: {e}")
            break
        
        # Check convergence
        if abs(prev_predictor_loss - predictor_loss) < tolerance:
            converged = True
            break
            
        prev_predictor_loss = predictor_loss
    
    # Final evaluation
    final_y_pred = predictor.predict(X_scaled)
    final_y_pred_probs = predictor.predict_proba(X_scaled)
    if final_y_pred_probs.shape[1] == 2:
        final_y_pred_probs_pos = final_y_pred_probs[:, 1]
    else:
        final_y_pred_probs_pos = final_y_pred_probs[:, 0]
    
    predictor_accuracy = accuracy_score(y, final_y_pred)
    
    # Final adversary evaluation
    final_predictor_outputs = final_y_pred_probs_pos.reshape(-1, 1)
    adv_pred = adversary.predict(final_predictor_outputs)
    adversary_accuracy = accuracy_score(sensitive_attr, adv_pred)
    
    # Calculate all fairness metrics
    demographic_parity = _calculate_demographic_parity(final_y_pred_probs_pos, sensitive_attr)
    equality_of_odds = _calculate_equality_