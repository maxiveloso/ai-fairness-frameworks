import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

class AdversarialEncoder(BaseEstimator, TransformerMixin):
    """Neural network encoder component for adversarial censoring."""
    
    def __init__(self, hidden_layers: Tuple[int, ...] = (50, 20), 
                 learning_rate: float = 0.001, max_iter: int = 200):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.encoder_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'AdversarialEncoder':
        # Use MLPRegressor as encoder to learn representations
        self.encoder_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            random_state=42
        )
        # Fit encoder using identity mapping for unsupervised learning
        self.encoder_.fit(X, X)
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        # Extract hidden representations from the encoder
        if self.encoder_ is None:
            raise ValueError("Encoder must be fitted before transform")
        
        # Get activations from the last hidden layer
        activations = X
        for i, (coef, intercept) in enumerate(zip(self.encoder_.coefs_[:-1], 
                                                  self.encoder_.intercepts_[:-1])):
            activations = np.maximum(0, np.dot(activations, coef) + intercept)  # ReLU activation
        
        return activations

def adversarial_censoring_of_representations(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    sensitive_attr: Union[np.ndarray, pd.Series],
    encoder_layers: Tuple[int, ...] = (100, 50),
    decoder_layers: Tuple[int, ...] = (50, 100),
    predictor_layers: Tuple[int, ...] = (50, 25),
    adversary_layers: Tuple[int, ...] = (50, 25),
    learning_rate: float = 0.001,
    n_iterations: int = 100,
    adversary_weight: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Implement Adversarial Censoring of Representations for fair machine learning.
    
    This technique uses a minimax optimization approach with four neural network components:
    - Encoder: Maps input features to learned representations
    - Decoder: Reconstructs original features from representations  
    - Predictor: Makes predictions on target variable from representations
    - Adversary: Tries to predict sensitive attributes from representations
    
    The main objective is to learn representations that are useful for the primary task
    but contain minimal information about sensitive attributes, promoting fairness.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input feature matrix
    y : array-like of shape (n_samples,)
        Target variable for main prediction task
    sensitive_attr : array-like of shape (n_samples,)
        Sensitive attribute to be censored from representations
    encoder_layers : tuple of int, default=(100, 50)
        Hidden layer sizes for encoder network
    decoder_layers : tuple of int, default=(50, 100)  
        Hidden layer sizes for decoder network
    predictor_layers : tuple of int, default=(50, 25)
        Hidden layer sizes for predictor network
    adversary_layers : tuple of int, default=(50, 25)
        Hidden layer sizes for adversary network
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer
    n_iterations : int, default=100
        Number of alternating optimization iterations
    adversary_weight : float, default=1.0
        Weight for adversarial loss in minimax objective
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'reconstruction_loss': Mean squared error for decoder reconstruction
        - 'prediction_accuracy': Accuracy/R² for main prediction task
        - 'adversary_accuracy': Accuracy for adversary predicting sensitive attribute
        - 'fairness_score': 1 - adversary_accuracy (higher is more fair)
        - 'representations': Learned representations from encoder
        - 'convergence_history': Training losses over iterations
        - 'model_components': Fitted encoder, decoder, predictor, adversary
        
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> 
    >>> # Generate synthetic data
    >>> X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    >>> sensitive = np.random.binomial(1, 0.3, 1000)  # Binary sensitive attribute
    >>> 
    >>> # Apply adversarial censoring
    >>> results = adversarial_censoring_of_representations(X, y, sensitive)
    >>> print(f"Fairness score: {results['fairness_score']:.3f}")
    >>> print(f"Prediction accuracy: {results['prediction_accuracy']:.3f}")
    """
    
    # Input validation
    X = np.asarray(X)
    y = np.asarray(y)
    sensitive_attr = np.asarray(sensitive_attr)
    
    if X.shape[0] != len(y) or X.shape[0] != len(sensitive_attr):
        raise ValueError("X, y, and sensitive_attr must have the same number of samples")
    
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
        
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive")
        
    if adversary_weight < 0:
        raise ValueError("adversary_weight must be non-negative")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X_scaled, y, sensitive_attr, test_size=test_size, random_state=random_state,
        stratify=sensitive_attr
    )
    
    n_features = X_train.shape[1]
    representation_dim = encoder_layers[-1]
    
    # Determine if target is classification or regression
    is_classification = len(np.unique(y)) <= 10 and np.all(y == y.astype(int))
    is_sensitive_binary = len(np.unique(sensitive_attr)) == 2
    
    # Initialize network components
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Encoder (learns representations)
        encoder = AdversarialEncoder(
            hidden_layers=encoder_layers,
            learning_rate=learning_rate,
            max_iter=50
        )
        
        # Decoder (reconstructs input from representations)
        decoder = MLPRegressor(
            hidden_layer_sizes=decoder_layers + (n_features,),
            learning_rate_init=learning_rate,
            max_iter=50,
            random_state=random_state
        )
        
        # Predictor (predicts target from representations)
        if is_classification:
            predictor = MLPClassifier(
                hidden_layer_sizes=predictor_layers,
                learning_rate_init=learning_rate,
                max_iter=50,
                random_state=random_state
            )
        else:
            predictor = MLPRegressor(
                hidden_layer_sizes=predictor_layers,
                learning_rate_init=learning_rate,
                max_iter=50,
                random_state=random_state
            )
        
        # Adversary (tries to predict sensitive attribute from representations)
        if is_sensitive_binary:
            adversary = MLPClassifier(
                hidden_layer_sizes=adversary_layers,
                learning_rate_init=learning_rate,
                max_iter=50,
                random_state=random_state
            )
        else:
            adversary = MLPRegressor(
                hidden_layer_sizes=adversary_layers,
                learning_rate_init=learning_rate,
                max_iter=50,
                random_state=random_state
            )
    
    # Training history
    history = {
        'reconstruction_loss': [],
        'prediction_loss': [],
        'adversary_loss': [],
        'total_loss': []
    }
    
    # Alternating minimax optimization
    for iteration in range(n_iterations):
        # Step 1: Fit encoder to learn representations
        encoder.fit(X_train)
        representations_train = encoder.transform(X_train)
        
        # Step 2: Train decoder for reconstruction
        decoder.fit(representations_train, X_train)
        X_reconstructed = decoder.predict(representations_train)
        reconstruction_loss = mean_squared_error(X_train, X_reconstructed)
        
        # Step 3: Train predictor for main task
        predictor.fit(representations_train, y_train)
        if is_classification:
            y_pred_train = predictor.predict(representations_train)
            prediction_loss = 1 - accuracy_score(y_train, y_pred_train)
        else:
            y_pred_train = predictor.predict(representations_train)
            prediction_loss = mean_squared_error(y_train, y_pred_train)
        
        # Step 4: Train adversary (maximize adversary loss in minimax game)
        adversary.fit(representations_train, s_train)
        if is_sensitive_binary:
            s_pred_train = adversary.predict(representations_train)
            adversary_loss = 1 - accuracy_score(s_train, s_pred_train)
        else:
            s_pred_train = adversary.predict(representations_train)
            adversary_loss = mean_squared_error(s_train, s_pred_train)
        
        # Total loss combines reconstruction, prediction, and adversarial terms
        total_loss = reconstruction_loss + prediction_loss - adversary_weight * adversary_loss
        
        # Store training history
        history['reconstruction_loss'].append(reconstruction_loss)
        history['prediction_loss'].append(prediction_loss)
        history['adversary_loss'].append(adversary_loss)
        history['total_loss'].append(total_loss)
    
    # Final evaluation on test set
    representations_test = encoder.transform(X_test)
    
    # Reconstruction performance
    X_test_reconstructed = decoder.predict(representations_test)
    final_reconstruction_loss = mean_squared_error(X_test, X_test_reconstructed)
    
    # Main task performance
    if is_classification:
        y_test_pred = predictor.predict(representations_test)
        prediction_accuracy = accuracy_score(y_test, y_test_pred)
    else:
        y_test_pred = predictor.predict(representations_test)
        prediction_accuracy = max(0, 1