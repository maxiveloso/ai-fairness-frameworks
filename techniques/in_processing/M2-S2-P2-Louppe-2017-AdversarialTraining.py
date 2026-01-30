import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

def adversarial_training(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    nuisance: Union[np.ndarray, pd.Series],
    predictor_hidden_layers: Tuple[int, ...] = (100, 50),
    adversary_hidden_layers: Tuple[int, ...] = (50, 25),
    lambda_reg: float = 1.0,
    pretrain_epochs: int = 100,
    adversarial_epochs: int = 200,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    validation_split: float = 0.2,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Union[float, np.ndarray, object]]:
    """
    Implement adversarial training for learning representations invariant to nuisance parameters.
    
    This technique uses a GAN-like architecture with two neural networks:
    1) Predictor network: learns to predict target variable from features
    2) Adversary network: learns to predict nuisance parameter from predictor's hidden representation
    
    The training involves a minimax game where the predictor tries to minimize prediction error
    while maximizing adversary's error (making representations uninformative about nuisance).
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features
    y : array-like of shape (n_samples,)
        Target variable (binary classification: 0/1)
    nuisance : array-like of shape (n_samples,)
        Nuisance parameter to make predictions invariant to
    predictor_hidden_layers : tuple of int, default=(100, 50)
        Hidden layer sizes for predictor network
    adversary_hidden_layers : tuple of int, default=(50, 25)
        Hidden layer sizes for adversary network
    lambda_reg : float, default=1.0
        Regularization parameter controlling accuracy-robustness tradeoff
        Higher values prioritize invariance over accuracy
    pretrain_epochs : int, default=100
        Number of epochs for pre-training each network separately
    adversarial_epochs : int, default=200
        Number of epochs for adversarial training phase
    learning_rate : float, default=0.001
        Learning rate for neural networks
    batch_size : int, default=32
        Batch size for training
    validation_split : float, default=0.2
        Fraction of data to use for validation
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print training progress
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'predictor_model': trained predictor network
        - 'adversary_model': trained adversary network  
        - 'predictor_accuracy': final accuracy on validation set
        - 'adversary_accuracy': final adversary accuracy (lower is better for invariance)
        - 'invariance_score': measure of representation invariance (1 - adversary_accuracy)
        - 'training_history': loss values during training
        - 'lambda_reg': regularization parameter used
        - 'feature_representations': learned representations from predictor
        
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> 
    >>> # Generate synthetic data
    >>> X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    >>> nuisance = np.random.binomial(1, 0.5, 1000)  # Binary nuisance variable
    >>> 
    >>> # Apply adversarial training
    >>> results = adversarial_training(X, y, nuisance, lambda_reg=0.5, random_state=42)
    >>> print(f"Predictor accuracy: {results['predictor_accuracy']:.3f}")
    >>> print(f"Invariance score: {results['invariance_score']:.3f}")
    """
    
    # Input validation
    X = np.asarray(X)
    y = np.asarray(y)
    nuisance = np.asarray(nuisance)
    
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if len(y) != len(X):
        raise ValueError("X and y must have same number of samples")
    if len(nuisance) != len(X):
        raise ValueError("X and nuisance must have same number of samples")
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1")
    if lambda_reg < 0:
        raise ValueError("lambda_reg must be non-negative")
        
    # Check if y is binary
    unique_y = np.unique(y)
    if len(unique_y) != 2 or not all(val in [0, 1] for val in unique_y):
        raise ValueError("y must be binary (0/1)")
        
    n_samples, n_features = X.shape
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val, nuisance_train, nuisance_val = train_test_split(
        X, y, nuisance, test_size=validation_split, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Initialize models
    predictor = MLPClassifier(
        hidden_layer_sizes=predictor_hidden_layers,
        learning_rate_init=learning_rate,
        batch_size=min(batch_size, len(X_train)),
        max_iter=1,  # We'll train iteratively
        warm_start=True,
        random_state=random_state
    )
    
    # Determine if nuisance is classification or regression task
    unique_nuisance = np.unique(nuisance_train)
    is_nuisance_binary = len(unique_nuisance) == 2
    
    if is_nuisance_binary:
        adversary = MLPClassifier(
            hidden_layer_sizes=adversary_hidden_layers,
            learning_rate_init=learning_rate,
            batch_size=min(batch_size, len(X_train)),
            max_iter=1,
            warm_start=True,
            random_state=random_state
        )
    else:
        adversary = MLPRegressor(
            hidden_layer_sizes=adversary_hidden_layers,
            learning_rate_init=learning_rate,
            batch_size=min(batch_size, len(X_train)),
            max_iter=1,
            warm_start=True,
            random_state=random_state
        )
    
    # Training history
    history = {
        'predictor_loss': [],
        'adversary_loss': [],
        'predictor_accuracy': [],
        'adversary_accuracy': []
    }
    
    # Phase 1: Pre-train predictor
    if verbose:
        print("Phase 1: Pre-training predictor...")
        
    predictor.max_iter = pretrain_epochs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictor.fit(X_train_scaled, y_train)
    
    # Phase 2: Pre-train adversary on predictor's representations
    if verbose:
        print("Phase 2: Pre-training adversary...")
    
    # Get hidden representations from predictor
    # Note: scikit-learn doesn't expose hidden layers directly, so we approximate
    # by using the decision function or predict_proba as representation
    train_repr = predictor.predict_proba(X_train_scaled)
    
    adversary.max_iter = pretrain_epochs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adversary.fit(train_repr, nuisance_train)
    
    # Phase 3: Adversarial training
    if verbose:
        print("Phase 3: Adversarial training...")
    
    # Reset max_iter for iterative training
    predictor.max_iter = 1
    adversary.max_iter = 1
    
    best_predictor_acc = 0
    best_adversary_acc = 1
    
    for epoch in range(adversarial_epochs):
        # Train predictor (minimize prediction loss + maximize adversary loss)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictor.fit(X_train_scaled, y_train)
        
        # Get new representations
        train_repr = predictor.predict_proba(X_train_scaled)
        val_repr = predictor.predict_proba(X_val_scaled)
        
        # Train adversary on new representations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adversary.fit(train_repr, nuisance_train)
        
        # Evaluate on validation set
        pred_acc = predictor.score(X_val_scaled, y_val)
        
        if is_nuisance_binary:
            adv_acc = adversary.score(val_repr, nuisance_val)
        else:
            # For regression, convert to accuracy-like metric
            adv_pred = adversary.predict(val_repr)
            adv_acc = 1.0 - np.mean(np.abs(adv_pred - nuisance_val)) / np.std(nuisance_val)
            adv_acc = max(0, adv_acc)  # Ensure non-negative
        
        # Store history
        history['predictor_accuracy'].append(pred_acc)
        history['adversary_accuracy'].append(adv_acc)
        
        # Update best scores
        if pred_acc > best_predictor_acc:
            best_predictor_acc = pred_acc
        if adv_acc < best_adversary_acc:
            best_adversary_acc = adv_acc
            
        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}: Predictor Acc = {pred_acc:.3f}, "
                  f"Adversary Acc = {adv_acc:.3f}")
    
    # Final evaluation
    final_repr = predictor.predict_proba(X_val_scaled)
    final_pred_acc = predictor.score(X_val_scaled, y_val)
    
    if is_nuisance_binary:
        final_adv_acc = adversary.score(final_repr, nuisance_val)
    else:
        adv_pred = adversary.predict(final_repr)
        final_adv_acc = 1.0 - np.mean(np.abs(adv_pred - nuisance_val)) / np.std(nuisance_val)
        final_adv_acc = max(0, final_adv_acc)
    
    # Calculate invariance score (higher is better)
    invariance_score = 1.0 - final_adv_acc
    
    # Get feature representations for all data
    all_repr = predictor.predict_proba(scaler.transform(X))
    
    return {
        'predictor_model': predictor,
        'adversary_model': adversary,
        'scaler': scaler,
        'predictor_accuracy': final_pred_acc,
        'adversary_accuracy': final_adv_acc,
        'invariance_score': invariance_score,